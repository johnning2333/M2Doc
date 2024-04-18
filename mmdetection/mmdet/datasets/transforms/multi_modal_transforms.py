import numpy as np

from typing import List, Optional, Sequence, Tuple, Union
import mmcv
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms import RandomFlip, Resize, RandomCrop
from mmdet.structures.bbox import autocast_box_type
from mmcv.image.geometric import _scale_size
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData, PixelData
from mmcv.transforms import to_tensor
from mmdet.structures.bbox import get_box_type
import torch
import json
import numpy as np
@TRANSFORMS.register_module()
class LoadAnnotationsMM(LoadAnnotations):
    def __init__(
        self,
        with_text: bool = False,
        **kwargs
    ) -> None:

        self.with_text = with_text
        super().__init__(**kwargs)

    def _load_text_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = results['text_bboxes']
        if len(gt_bboxes) == 0:
            results['text_bboxes'] = None
        else:
            if self.box_type is None:
                results['text_bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
            else:
                _, box_type_cls = get_box_type(self.box_type)
                results['text_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        
    def _load_content(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        paths = results['content_path']
        if len(paths) == 0:
            results['text_bboxes'] = None
            results['texts'] = None
        else:
            ann = json.load(open(paths))
            content = ann.get("content_ann", None)
            text_bboxes = content['bboxes']
            texts = content['texts']
            if self.box_type is None:
                results['text_bboxes'] = np.array(
                    text_bboxes, dtype=np.float32)
                results['texts'] = texts
            else:
                _, box_type_cls = get_box_type(self.box_type)
                results['text_bboxes'] = box_type_cls(text_bboxes, dtype=torch.float32)
                results['texts'] = texts
        
    
    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        if self.with_text:
            self._load_text_bboxes(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'(with_text={self.with_text}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

@TRANSFORMS.register_module()
class RandomFlipMM(RandomFlip):
    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip text_bboxes
        if results.get('text_bboxes', None) is not None:
            results['text_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)

@TRANSFORMS.register_module()
class ResizeMM(Resize):
    def _resize_text_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('text_bboxes', None) is not None:
            results['text_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['text_bboxes'].clip_(results['img_shape'])

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_text_bboxes(results)
        self._record_homography_matrix(results)
        return results

@TRANSFORMS.register_module()
class RandomCropMM(RandomCrop):
    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

        if results.get('text_bboxes', None) is not None:
            bboxes = results['text_bboxes']
            texts = results['texts']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            if not valid_inds.any():
                results['text_bboxes'] = None
                results['texts'] = None
            else:
                results['text_bboxes'] = bboxes[valid_inds]
                results['texts'] = [texts[i] for i,idx in enumerate(valid_inds) if idx == 1]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results

@TRANSFORMS.register_module()
class PackDetInputsMM(PackDetInputs):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
    }

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        # print(results.keys())
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        
        text_instance_data = InstanceData()
        if 'text_bboxes' in results and results['text_bboxes'] is not None:
            valid_idx = [i for i in range(len(results['text_bboxes']))]
            text_instance_data['texts'] = results['texts']
            text_instance_data['text_bboxes'] = results['text_bboxes'][valid_idx]
            data_sample.text_instances = text_instance_data

        data_sample.text_instances = text_instance_data

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results