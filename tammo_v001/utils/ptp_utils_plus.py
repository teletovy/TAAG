'''
가칭 v001로 추후의 개선 idea 구현 용 ptp_utils.py attention store와 attention processor용 code ptp 기반 - 0916
앞으로의 base attention store - 0918
'''
from typing import List, Tuple
import numpy as np
import torch
from diffusers.models.attention_processor import Attention
from torch_kmeans import KMeans
from .attn_utils import fn_smoothing_func, fn_get_topk
import nltk

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        '''
        key 기반으로 unet의 attention maps를 저장하는 용도
        '''
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        if self.cur_att_layer >= 0:            
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[key].append(attn)
            # NOTE 0916 시작 version은 16x16 resolution map만 사용하는 것부터 
            # 아래는 다양한 resolution map을 이용할 때 필요
            # if (place_in_unet == "mid" and attn.shape[1] == np.prod(self.attn_res[0])) or\
            #     (place_in_unet == "up" and attn.shape[1] == np.prod(self.attn_res[1])) or\
            #     (key == 'down_self' and attn.shape[1] == np.prod(self.attn_res[1])):
            #     self.step_store[key].append(attn)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
    
    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention
    
    def get_average_seg_self_attention(self):
        average_attention = self.seg_self_attention_store
        return average_attention
    
    def aggregate_attention(self, from_where: List[str], res: Tuple, is_cross: bool = True) -> torch.Tensor:
        """
        Aggregates the attention across the different layers and heads at the specified resolution.
        NOTE 0916 또한, 앞으로는 resolution도 인자로 입력 받아 resolution 별 aggregation
        """
        out = []
        attention_maps = self.get_average_attention() # attention dict
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                cross_maps = item.reshape(-1, res[0], res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out
    
    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

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
        self.attn_res = attn_res # NOTE not only just one resolution like (16, 16) but also [(16, 16), (32, 32)]
        # NOTE 0916 self.seg is deleted
        self.mask = None # forward guidance를 위한 변수


class AttendExciteAttnProcessor:
    #NOTE 09/23 new from BA
    EPSILON = 1e-5
    FILTER_TAGS = {
        'CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WRB'}
    TAG_RULES = {'left': 'IN', 'right': 'IN', 'top': 'IN', 'bottom': 'IN'}

    def __init__(self, attnstore, place_in_unet, 
                token_indices,
                tokenizer, # pipeline
                prompts, # List[str]
                # K=None,
                head_dim=8
                 ):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet        
        self.token_indices = token_indices #NOTE 09/23 new from BA        
        self.subject_token_indices = [] #NOTE 09/23 new from BA        
        for token_idx in token_indices:
            self.subject_token_indices.append([token_idx]) #NOTE 09/23 new from BA        
        
        self.head_dim = head_dim
        # self.optimized = False #NOTE new from BA impl, masking을 위한 flag #NOTE 09/23 firstly fg is done always
        self.tokenizer = tokenizer#NOTE 09/23 new ; BA reimpl
        self.prompts = prompts#NOTE 09/23 new ; BA reimpl
        self._determine_filter_tokens() #NOTE 09/23 new from BA     
        #NOTE 09/24 maybe deprecated, this part move into pipelines
        # self.eos_token_index = None
        # self._determine_eos_token() 

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

        if self.attnstore.mask is not None and hidden_states.shape[1] == np.prod(self.attnstore.attn_res):            
            #NOTE original query 16x256x160, key 16x256x160 or 16x77x160
            # attention_probs = self.get_fg_attention_score(query, key, attn, attention_mask, is_cross)
            if attn.upcast_attention:
                query = query.float()
                key = key.float()

            if attention_mask is None:
                baddbmm_input = torch.empty(
                    query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                )
                beta = 0
            else:
                baddbmm_input = attention_mask
                beta = 1

            attention_scores = torch.baddbmm(
                baddbmm_input,
                query,
                key.transpose(-1, -2),
                beta=beta,
                alpha=attn.scale,
            )
            del baddbmm_input
            if attn.upcast_softmax:
                attention_scores = attention_scores.float() # head * batch x i x l
            #NOTE 09/23 AttendExciteAttnProcessor.get_manual_attention_scores() got multiple values for argument 'q' 때문에 closed
            # attn_maps_bef_sofmax -> attention_scores
            # attn_maps_bef_sofmax = self.get_manual_attention_scores(q=query, k=key, attention_mask=attention_mask, attention_module=attn) #TODO should check dim = b*h x i x l         

            if is_cross:
                fg_mask = self.get_fg_cross_mask(attention_maps=attention_scores, dtype=query.dtype) # batch x 1 x i x l
            else:
                fg_mask = self.get_fg_self_mask(attention_maps=attention_scores, dtype=query.dtype) # batch x 1 x i x l
            
            # b x h x i x l
            attention_probs = attention_scores.reshape(batch_size , self.head_dim, query.size(1), key.size(1)) + fg_mask
            attention_probs = attention_probs.reshape(-1, query.size(1), key.size(1)).softmax(-1)
            attention_probs = attention_probs.to(query.dtype)

        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask) # original        
            
        # only need to store attention maps during the Attend and Excite process
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)        
        
        #NOTE while cfg attention_probs 2 * #head x i x l, if not 1*#head x i x l
        hidden_states = torch.bmm(attention_probs, value) 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
    #NOTE 09/23 new implementation of fg from BA -> get_fg_cross_mask, get_fg_self_mask function
    def get_fg_cross_mask(self, attention_maps, dtype):        
        b, i, l = attention_maps.shape  # b * h x i x l
        # b = b // self.head_dim                
        attention_maps = attention_maps.reshape(b // self.head_dim, self.head_dim, i, l)
        
        subject_masks = []
        for token_idx, mask in self.attnstore.mask.items():
            mask = torch.from_numpy(mask).bool().reshape(-1).to(attention_maps.device)
            mask = mask.unsqueeze(0)
            subject_masks.append(mask)
        subject_masks = torch.cat(subject_masks, dim=0).unsqueeze(0) #NOTE should be b x #S x i
        assert subject_masks.shape == (1, len(self.attnstore.mask.keys()), i) # 1 x #S x i
        
        bg_mask = self.get_background_mask(device=attention_maps.device) #NOTE should be 1 x i or b x i
        min_value = torch.finfo(dtype).min        
        
        # include_background = self.optimized or (not self.mask_cross_during_guidance and self.cur_step < self.max_guidance_iter_per_step) #NOTE 09/23 일단 always include bg ! ; BA do same
        subject_masks = torch.logical_or(subject_masks, bg_mask.unsqueeze(1)) #NOTE include background, should be b x s# x i
        sim_masks = torch.zeros((1, i, l), dtype=dtype, device=attention_maps.device) #NOTE multi batch 고려 안함, 원본 code 또한
        for token_indices in (*self.subject_token_indices, self.filter_token_indices):
            sim_masks[:, :, token_indices] = min_value
        
        for batch_index in range(1): #NOTE 1로 fix
            for subject_mask, token_indices in zip(subject_masks[batch_index], self.subject_token_indices):
                for token_index in token_indices:
                    sim_masks[batch_index, subject_mask, token_index] = 0

        #NOTE 09/23 include_background always로 보임 -> my case에 대해서 다르게 한번 해볼 필요가 있을 수도? #NOTE 09/23 일단 always include bg !
        # if self.mask_eos and not include_background:
        #     for batch_index, background_mask in zip(range(b), bg_mask):
        #         sim_masks[batch_index, background_mask, self.eos_token_index] = min_value

        if b // self.head_dim == 2:
            return torch.cat((torch.zeros_like(sim_masks), sim_masks)).unsqueeze(1)
        elif b // self.head_dim == 1:
            return sim_masks.unsqueeze(1)
        else:
            ValueError(f'Invalid batch size {b}')
    
    def get_fg_self_mask(self, attention_maps, dtype):
        #TODO should check all process below
        device = attention_maps.device
        b, i, l = attention_maps.shape
        assert i == l
        attention_maps = attention_maps.reshape( b//self.head_dim, self.head_dim, i, l)
        _, _, i, l = attention_maps.shape
        #NOTE 09/24 mask 만드는 batch_size = 1로 고정 TODO cross mask도 고칠것 고치고 이 주석은 지울것
        # if batch_size != 1:
        #     batch_size = batch_size // 2

        # BA format에 맞춰 mask re build
        subject_masks = []
        for token_idx, mask in self.attnstore.mask.items():
            mask = torch.from_numpy(mask).bool().reshape(-1).to(attention_maps.device)
            mask = mask.unsqueeze(0)
            subject_masks.append(mask)
        subject_masks = torch.cat(subject_masks, dim=0).unsqueeze(0) #NOTE should be b x #S x i
        assert subject_masks.shape == (1, len(self.attnstore.mask.keys()), i) # 1 x #S x i
        
        bg_mask = self.get_background_mask(device=attention_maps.device) #NOTE should be should be i

        min_value = torch.finfo(dtype).min
        sim_masks = torch.zeros((1, i, l), dtype=dtype, device=device)  # b i j NOTE 이때 b = negative prompt를 제외한 batch size
        for batch_index, background_mask in zip(range(1), bg_mask):
            sim_masks[batch_index, ~background_mask, ~background_mask] = min_value

        for batch_index in range(1):
            for subject_mask in subject_masks[batch_index]:
                subject_sim_mask = sim_masks[batch_index, subject_mask]
                condition = torch.logical_or(subject_sim_mask == 0, subject_mask.unsqueeze(0))
                sim_masks[batch_index, subject_mask] = torch.where(condition, 0, min_value).to(dtype=dtype)

        # at this line sim_masks should be h x i x l? -> check BA
        if b // self.head_dim == 2:
            return torch.cat((sim_masks, sim_masks)).unsqueeze(1) #NOTE should be b x h x i x l
        elif b // self.head_dim == 1:            
            return sim_masks.unsqueeze(1)
        else:
            ValueError(f'Invalid batchsize {b}')
    
    #TODO NOTE 09/23 old ver fg는 폐기 이전 src 이용해서 다시 복구 할 것 or old ver 참고    

    def get_background_mask(self, device):

        masks_list = []
        for idx, v in self.attnstore.mask.items():
            v = torch.from_numpy(v).bool()
            h, w = v.shape
            masks_list.append(v.reshape(-1))
        masks_list = torch.stack(masks_list, dim=0) # NOTE should check s x hw
        bg_mask = masks_list.sum(dim=0) == 0 # (hw)
        # bg_mask = bg_mask.reshape(h, w).to(device) # h x w
        return bg_mask.unsqueeze(0).to(device)
    
    #NOTE 09/23 new from BA
    def _tokenize(self):
        ids = self.tokenizer.encode(self.prompts[0])
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        return [token[:-4] for token in tokens]  # remove ending </w>

    def _tag_tokens(self):
        tagged_tokens = nltk.pos_tag(self._tokenize())
        return [type(self).TAG_RULES.get(token, tag) for token, tag in tagged_tokens]
    
    def _determine_filter_tokens(self):        
        tags = self._tag_tokens()
        self.filter_token_indices = [i + 1 for i, tag in enumerate(tags) if tag in type(self).FILTER_TAGS]
        # print(f'{self.filter_token_indices}')

    #NOTE 09/24 maybe deprecated, this part move into pipelines
    # def _determine_eos_token(self):
    #     tokens = self._tokenize()
    #     eos_token_index = len(tokens) + 1
    #     if self.eos_token_index is None:
    #         self.eos_token_index = eos_token_index
    #     elif eos_token_index != self.eos_token_index:
    #         raise ValueError(f'Wrong EOS token index. Tokens are: {tokens}.')


    #NOTE 09/23 new for BA ver // 09/24 deprecated
    # def get_manual_attention_scores(q, k, attention_mask, attention_module):        
        
    #     if attention_module.upcast_attention:
    #         q = q.float()
    #         k = k.float()

    #     if attention_mask is None:
    #         baddbmm_input = torch.empty(
    #             q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device
    #         )
    #         beta = 0
    #     else:
    #         baddbmm_input = attention_mask
    #         beta = 1

    #     attention_scores = torch.baddbmm(
    #         baddbmm_input,
    #         q,
    #         k.transpose(-1, -2),
    #         beta=beta,
    #         alpha=attention_module.scale,
    #     )
    #     del baddbmm_input
    #     if attention_module.upcast_softmax:
    #         attention_scores = attention_scores.float()
    #     # attention_probs = attention_scores.softmax(dim=-1)
    #     # del attention_scores
    #     # attention_probs = attention_probs.to(dtype)
    #     return attention_scores

    

