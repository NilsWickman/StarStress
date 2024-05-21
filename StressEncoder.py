import torch
import torch.nn as nn

from typing import Optional, Any, Union, Callable

import torch
from torch.nn import functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm
from typing import Optional, Tuple
from self_attention_forward import stress_sa_head_forward
from torch.nn.functional import linear

LOAD = True
TRAIN = False
LOAD_PATH = "stress_model/Epoch510Threat_weights.tm"

class StressFormer (nn.TransformerEncoderLayer):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first,
                bias, device, dtype)
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
        #                                    bias=bias, batch_first=batch_first,
        #                                    **factory_kwargs)
        

        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)

    def forward(
            self,
            src: Tensor,
            tot_result,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tot_result, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, tot_result, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x: Tensor, tot_result,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x, tot_result,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)
    
    
class SelfAttention (Module):

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, layer_norm_eps: float = 1e-5,) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.norm = LayerNorm(embed_dim, eps=layer_norm_eps, bias=bias, **factory_kwargs)


        if LOAD:
            checkpoint = torch.load(LOAD_PATH)
            self.epoch = checkpoint['epoch']
            self.threat_weight = checkpoint['Threat Weights']
            print("Loaded Stress Weights")
        else:
            self.epoch = 0
            with torch.no_grad():
                self.threat_weight = torch.zeros((embed_dim, embed_dim), **factory_kwargs, requires_grad=True)
                #self.threat_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))

        self.new_threat_weight = self.threat_weight
        device = torch.device("cuda:0")
        self.step = 0
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            tot_result,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:


        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path.")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        

        attn_output = stress_sa_head_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal)

        if TRAIN == True:
            self.threat_weight.data = update_threat_weights(self.threat_weight, attn_output, tot_result).data
        
        threat = query 

        Et = threat.size(-1)
        assert self.threat_weight.shape == (Et, Et), f"expecting query weights shape of {(Et, Et)}, but got {Et.shape}"

        threat = linear(threat.to(query.device), self.threat_weight.to(query.device), None)

        attn_output = stress(attn_output, threat)
        if TRAIN == True:
            self.step += 1
            if self.step == 128:
                print("Weights: ", torch.mean(self.threat_weight))


                print("SAVING STRESS")
                self.epoch += 1
                path = "stress_model/Threat_weights.tm"
                torch.save(
                        {
                            'Threat Weights': self.new_threat_weight,
                            'epoch': self.epoch,
                        }, path)
                self.step = 0
                if self.epoch % 10 == 0:
                    path = "stress_model/Epoch" + str(self.epoch) + "Threat_weights.tm"
                    torch.save(
                            {
                                'Threat Weights': self.new_threat_weight,
                                'epoch': self.epoch,
                            }, path)
                print("SAVED STRESS")

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0)
        else:
            return attn_output
        


    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type
    
def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        return any(type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode for x in torch_dispatch_mode_stack)
    else:
        return False
    
def stress_activation(x):
    #mask = x > 1
    #mask2 = x < 1
    return torch.where(x > 0.2, x*x, torch.tensor(0.0))

def update_threat_weights(weights, attention, tot_result):
    tot_result = tot_result / 32
    attention = attention.to(weights.device)
    attention = attention * tot_result
    attention = torch.sum(attention, dim=1)
    weights = weights * 0.9995
    weights = weights + (attention * 0.0000001)
    return weights

def stress(attention, threat):
    difference =  threat - attention
    no_variation_threashold = 0.1

    negligible_mask = torch.abs(difference) <= no_variation_threashold
    positive_mask = difference > 0
    negative_mask = difference < 0

    stress = torch.where(negligible_mask, attention * 2, 
                          torch.where(positive_mask, attention + difference, attention - torch.abs(difference)))
    
    stress = torch.where(negative_mask, attention - difference, attention + torch.abs(difference))
    return stress

