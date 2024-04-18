# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union, Optional
import torch
from torch import Tensor

from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import OptConfigType
from mmdet.registry import MODELS
from mmdet.models.detectors.dino import DINO
ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
from mmdet.models.layers import (DeformableDetrTransformerDecoder, DeformableDetrTransformerEncoder,
                                 SinePositionalEncoding)
from mmdet.models.layers.transformer.utils import coordinate_to_encoding, inverse_sigmoid
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

from torch import nn
from ..layers.grid_layers import GridLayers
from mmdet.models.layers.transformer.utils import MLP

class DinoTransformerDecoderWithLateFusion(DeformableDetrTransformerDecoder):
    """Transformer encoder of DINO."""
    def __init__(self,
                 with_late_fusion=True,
                 grid_dims=64,
                 gamma = 0.5,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.with_late_fusion = with_late_fusion
        self.grid_dims = grid_dims
        self.gamma = gamma

        if self.with_late_fusion:
            self.line_proj = nn.Parameter(torch.empty(self.grid_dims, self.embed_dims))
            nn.init.xavier_uniform_(self.line_proj)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def calculate_iou(self, text_bboxes, rois):
        """
        Calculates the intersection over union (IOU) between two sets of bounding boxes.
        
        Parameters:
        text_bboxes (torch.Tensor): Tensor of text bounding boxes, with shape (num_text_boxes, 4).
                                    Each row contains the coordinates of a bounding box in the format (xmin, ymin, xmax, ymax).
        rois (torch.Tensor): Tensor of layout bounding boxes, with shape (num_layout_boxes, 4).
                                    Each row contains the coordinates of a bounding box in the format (xmin, ymin, xmax, ymax).
        
        Returns:
        torch.Tensor: 2D tensor of IOU values, with shape (num_text_boxes, num_layout_boxes).
        """
        # Compute the areas of all bounding boxes
        text_bboxes = torch.tensor(text_bboxes.cpu().numpy()).to(rois.device)
        text_bbox_area = (text_bboxes[:, 2] - text_bboxes[:, 0]) * (text_bboxes[:, 3] - text_bboxes[:, 1])
        layout_bbox_area = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
        
        # Compute the coordinates of the intersection boxes
        xmin = torch.max(text_bboxes[:, None, 0], rois[None, :, 0])
        ymin = torch.max(text_bboxes[:, None, 1], rois[None, :, 1])
        xmax = torch.min(text_bboxes[:, None, 2], rois[None, :, 2])
        ymax = torch.min(text_bboxes[:, None, 3], rois[None, :, 3])
        
        # Compute the area of the intersection boxes
        intersection_area = torch.clamp(xmax - xmin, min=0) * torch.clamp(ymax - ymin, min=0)
        
        # Compute the IOU values
        union_area = text_bbox_area[:, None] + layout_bbox_area[None, :] - intersection_area
        iou = intersection_area / text_bbox_area.view(-1, 1)
        
        return iou


    def pred_bboxes_with_late_fusion(self, refpoints, text_bboxes, text_embed, img_meta, iou_threash=0.5):
        """
            refpoint [n, 4]
        """
        img_shape = img_meta['img_shape']
        det_bboxes = bbox_cxcywh_to_xyxy(refpoints)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        layout_embeds = torch.zeros([refpoints.shape[0], self.embed_dims], device=refpoints.device)
        ious = self.calculate_iou(text_bboxes, det_bboxes)

        sen_masks = (ious>iou_threash).sum(0) != 0

        valid_layout_embeds =  (ious>iou_threash)[:, sen_masks].float().transpose(1,0) @ text_embed
        layout_embeds[sen_masks] = valid_layout_embeds @ self.line_proj
        return layout_embeds
    
    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                batch_data_samples: OptSampleList = None, **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (num_queries, bs, dim)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            if self.with_late_fusion:
                if lid==5:
                    refpoints = reference_points_input[:, :, 0, :]
                    bs, num, c = refpoints.shape
                    
                    batch_text_features = torch.zeros([bs, num, self.embed_dims], device=refpoints.device)
                    for i in range(bs):
                        sen_embeds = batch_data_samples[i].sen_embeds
                        img_meta = batch_data_samples[i].metainfo
                        if sen_embeds is not None:
                            text_bboxes = batch_data_samples[i].text_instances.text_bboxes
                            batch_text_features[i] = self.pred_bboxes_with_late_fusion(refpoints[i],
                                                    text_bboxes,
                                                    sen_embeds,
                                                    img_meta=img_meta)
                    mm_query = batch_text_features * self.gamma + query
                    mm_query = self.norm(mm_query)
                else:
                    mm_query = self.norm(query)

                query = layer(
                    mm_query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)
            else:
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)
            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


@MODELS.register_module()
class DINO_w_M2Doc(DINO):

    def __init__(self, grid_layers :OptConfigType = None,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        #sentence grid head
        if grid_layers is not None:
            # self.grid_head = MODELS.build(sentence_head)
            self.grid_head = GridLayers(**grid_layers)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoderWithLateFusion(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)


    def get_grid_input(self, batch_inputs, batch_data_samples):
        sengrid, batch_sen_embed = self.grid_head(batch_inputs, batch_data_samples)

        return sengrid, batch_sen_embed

    
    def extract_feat_w_early_fusion(self, batch_inputs: Tensor, batch_grids: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs, batch_grids)
        if self.with_neck:
            x = self.neck(x)
        return x    

    def extract_feat_wo_early_fusion(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone.forward_wo_early_fusion(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        sengrid, batch_text_embeddings = self.get_grid_input(batch_inputs, batch_data_samples)
        if sengrid is not None:
            img_feats = self.extract_feat_w_early_fusion(batch_inputs, sengrid)
        else:
            img_feats = self.extract_feat_wo_early_fusion(batch_inputs)
        for i in range(batch_inputs.shape[0]):
            if batch_text_embeddings is not None:
                batch_data_samples[i].sen_embeds = batch_text_embeddings[i]
            else:
                batch_data_samples[i].sen_embeds = None
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_inputs_dict['batch_data_samples'] = batch_data_samples

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        batch_data_samples: OptSampleList = None,) -> Dict:
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            batch_data_samples=batch_data_samples)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        sengrid, batch_text_embeddings = self.get_grid_input(batch_inputs, batch_data_samples)
        if sengrid is not None:
            img_feats = self.extract_feat_w_early_fusion(batch_inputs, sengrid)
        else:
            img_feats = self.extract_feat_wo_early_fusion(batch_inputs)
        for i in range(batch_inputs.shape[0]):
            if batch_text_embeddings is not None:
                batch_data_samples[i].sen_embeds = batch_text_embeddings[i]
            else:
                batch_data_samples[i].sen_embeds = None
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        batch_data_samples = self.add_pred_to_datasample(
        batch_data_samples, results_list)

        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        sengrid, batch_text_embeddings = self.get_grid_input(batch_inputs, batch_data_samples)
        if sengrid is not None:
            img_feats = self.extract_feat_w_early_fusion(batch_inputs, sengrid)
        else:
            img_feats = self.extract_feat_wo_early_fusion(batch_inputs)
        for i in range(batch_inputs.shape[0]):
            if batch_text_embeddings is not None:
                batch_data_samples[i].sen_embeds = batch_text_embeddings[i]
            else:
                batch_data_samples[i].sen_embeds = None
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        # print(len(data_samples[0].text_instances))
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')