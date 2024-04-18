
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from mmengine.model import BaseModule


class GridLayers(BaseModule):
    """Generate sentence embedding using pretrained language model, e.g. BERT
    """
    def __init__(self,
                 auto_model_path,
                 embedding_dim=768,
                 freeze_params=True,
                 batch_max_num=128,
                 select_layers=12,
                 max_token_nums=48,
                 with_line_proj=True,
                 ):
        """
        Args：
            auto_model_path (str): path to pretrained language model (e.g. BERT)
            embedding_dim (int): dim of input features
            freeze_params (boolean): whether to freeze params of pretrained language model, default to True.
            batch_max_num (int): the max num of texts in a batch due to memory limit, default 128.
        """
        self.auto_model_path = auto_model_path
        super().__init__()

        assert auto_model_path is not None
        self.batch_max_num = batch_max_num
        self.embedding_dim = embedding_dim
        
        ## load pretrained language model
        self.automodel = AutoModel.from_pretrained(auto_model_path, num_hidden_layers=select_layers)
        self.autotokenizer = AutoTokenizer.from_pretrained(auto_model_path, num_hidden_layers=select_layers)

        self.with_line_proj = with_line_proj
        if with_line_proj:
            self.line_proj = nn.Parameter(torch.empty(self.automodel.config.hidden_size, embedding_dim))
            nn.init.xavier_uniform_(self.line_proj)
        self.se_layers = select_layers
        self.max_token_nums = max_token_nums
        self.freeze_ = freeze_params
        self._freeze_automodel()

    def forward(self,
                img,
                batch_data_samples,
                ):
        """ Forward computation

        Args:
            img (Tensor): in shape of [B x C x H x W].
            gt_bboxes (list(Tensor)): bboxes for each text line in each image.
            gt_texts (list(list)): text contents for each image.
        Returns:
            Tensor: generated grid embedding maps in shape of [B x D x H x W], where D is the embedding_dim.
        """

        device = img.device
        batch_b, _, batch_h, batch_w = img.size()
        chargrid_map = img.new_full((batch_b, self.embedding_dim, batch_h, batch_w), 0)
        batch_text_embeddings = []


        for iter_b in range(batch_b):
            if len(batch_data_samples[iter_b].text_instances)==0:
                batch_text_embeddings.append(None)
                chargrid_map = None
                self.single_text_embedding = None
                continue
            gt_texts = batch_data_samples[iter_b].text_instances.texts
            gt_bboxes = batch_data_samples[iter_b].text_instances.text_bboxes
            
            per_img_texts = gt_texts
            start_idx = 0
            
            #single batch text embeddings shape [L x Hidden-Size]
            single_text_embedding = torch.zeros([len(per_img_texts), self.automodel.config.hidden_size], device=device)
    
            while start_idx < len(per_img_texts):
                max_length = min(start_idx+self.batch_max_num, len(per_img_texts))
                per_batch_texts = per_img_texts[start_idx: max_length]
                inputs = self.autotokenizer(per_batch_texts, return_tensors='pt',
                 padding=True, truncation=True, max_length=self.max_token_nums)
                inputs.to(device)
                outputs = self.automodel(**inputs)

                pooler_output = outputs['pooler_output']

                ##padding text_embeddings into single text embeddings
                single_text_embedding[start_idx:max_length] = pooler_output
                start_idx = max_length

            if self.with_line_proj:
                single_text_embedding = single_text_embedding @ self.line_proj

            batch_text_embeddings.append(single_text_embedding)
            for iter_b_l in range(single_text_embedding.shape[0]):
                w_start, h_start, w_end, h_end = gt_bboxes.cpu().numpy().round().astype(np.int_)[iter_b_l].tolist()
                chargrid_map[iter_b, :, h_start: h_end, w_start: w_end] = single_text_embedding[iter_b_l,
                                                                        :self.embedding_dim, None, None]
                                                                      
        return chargrid_map, batch_text_embeddings

    def _freeze_automodel(self):
        """Freeze params of the pretrained language models.
        """
        for _, param in self.automodel.named_parameters():
            if not self.freeze_:
                if not (_.startswith('encoder.layer.{}'.format(self.se_layers - 1))):
                    param.requires_grad = False
                else:
                    print(_)
            else:
                param.requires_grad = False
