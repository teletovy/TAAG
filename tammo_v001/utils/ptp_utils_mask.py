# import inspect
# import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from diffusers.models.attention_processor import Attention
# from torch_kmeans import KMeans
from .attn_utils import fn_smoothing_func, fn_get_topk

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.cur_att_layer >= 0:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[key].append(attn)
            
            # if attn.shape[1] == np.prod(self.attn_res) * 4 and (is_cross == False):
            #     self.seg_self_store[key].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()
        # if self.self_seg:
        #     self.seg_self_attention_store = self.seg_self_store
        #     self.seg_self_store = self.get_empty_store()


    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention
    
    def get_average_seg_self_attention(self):
        average_attention = self.seg_self_attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str], is_cross: bool = True) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        # NOTE 0829
        if self.self_seg:
            self.seg_self_store = self.get_empty_store()
            self.seg_self_attention_store = {}

    def __init__(self, attn_res, self_seg=False):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res # NOTE from now [8, 16]
        # NOTE 0829 TODO 0910 이제 필요없는 부분?
        # self.self_seg = self_seg
        # if self.self_seg:
        #     self.seg_self_store = self.get_empty_store()
        #     self.seg_self_attention_store = {}
        self.mask = None


class AttendExciteAttnProcessor:
    def __init__(self, attnstore, place_in_unet, 
                token_indices=None,
                K=None,
                head_dim=8
                 ):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        # NOTE 0811 new        
        self.token_indices = token_indices
        self.K = K
        self.head_dim = head_dim

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)        
        
        if self.attnstore.mask is not None and \
            is_cross and\
            hidden_states.shape[1] == np.prod(self.attnstore.attn_res):
            attention_probs = self.manual_get_attention_scores(query, key, attn, attention_mask, self.attnstore.mask)
        else:
            attention_probs = self.manual_get_attention_scores(query, key, attn, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)        
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states    
    
    # NOTE 0811 new
    # sa ca map detach clone 필요할 수도 !
    def get_attn_mask(self, self_attn_maps, cross_attn_maps, smooth_attentions=False):
        aggregated_cross_attn_maps = []
        aggregated_self_attn_maps = []
        self_attn_mask_list = []
        H, W = self.attnstore.attn_res
        target_self_mask = {}     
        
        for item in cross_attn_maps:
            cross_maps = item.reshape(-1, H, W, item.shape[-1])
            aggregated_cross_attn_maps.append(cross_maps)
        aggregated_cross_attn_maps = torch.cat(aggregated_cross_attn_maps, dim=0)
        aggregated_cross_attn_maps = aggregated_cross_attn_maps.sum(0) / aggregated_cross_attn_maps.shape[0]

        for item in self_attn_maps:
            self_maps = item.reshape(-1, H, W, item.shape[-1])
            aggregated_self_attn_maps.append(self_maps)
        aggregated_self_attn_maps = torch.cat(aggregated_self_attn_maps, dim=0)
        aggregated_self_attn_maps = aggregated_self_attn_maps.sum(0) / aggregated_self_attn_maps.shape[0]

        for index in self.token_indices:
            target_cross_attn_maps = aggregated_cross_attn_maps[:, :, index]
            if smooth_attentions:
                target_cross_attn_maps = fn_smoothing_func(target_cross_attn_maps)
            topk_coord_list, topk_value = fn_get_topk(target_cross_attn_maps, K=self.K)
            for coord_x, coord_y in topk_coord_list:
                self_attn_mask = aggregated_self_attn_maps[coord_x, coord_y]
                self_attn_mask_list.append(self_attn_mask)
            
            target_self_mask[index] = sum(self_attn_mask_list) / len(self_attn_mask_list)
        
        return target_self_mask

    
    # NOTE 0812 new because of gradient error, ref from diffusers attention module
    def manual_get_attention_scores(self, query, key, attention_module, attention_mask=None, multiplied_masks=None):
        
        dtype = query.dtype
        
        # i = query.shape[1]
        # d = key.shape[1]
        
        if attention_module.upcast_attention:
            query = query.float()
            key = key.float()
        
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask.clone()
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attention_module.scale,
        )
        del baddbmm_input

        if attention_module.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_scores_clone = attention_scores.clone() # 

        if multiplied_masks is not None:
            # multiplied_mask = {token idx : mask}
            b, i, l = attention_scores_clone.shape # b * h x i x l
            b = b // self.head_dim
            attention_scores_clone = attention_scores_clone.reshape(b, self.head_dim, i, l)
            # text_cond_attention_scores_clone = attention_scores_clone[1]
            for token_idx, mask in multiplied_masks.items():
                mask = torch.from_numpy(mask).to(attention_scores_clone.device)
                mask = mask.reshape(-1, i).squeeze(0) # 1 x i
                attention_scores_clone[1, :, :, token_idx] = attention_scores_clone[1, :, :, token_idx] * mask

            attention_scores_clone = attention_scores_clone.reshape(-1, i, l)

        # attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.nn.functional.softmax(attention_scores_clone, dim=-1)
        # del attention_scores
        del attention_scores_clone

        attention_probs = attention_probs.to(dtype)

        return attention_probs.clone()


    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    