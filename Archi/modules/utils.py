from typing import Dict, Any, Optional, List, Callable, Union

import torch
import torch.nn as nn

from functools import partial
import copy

import Archi 


def load_module(module_key, module_kwargs):
    module_cls = getattr(Archi.modules, module_key, None)
    if module_cls is None:
        raise NotImplementedError
    module = module_cls(**module_kwargs)

    print(module)
    return module 
   
def layer_init(layer, w_scale=1.0, nonlinearity='relu'):
    for name, param in layer._parameters.items():
        if param is None or param.data is None: continue
        if 'bias' in name:
            #layer._parameters[name].data.fill_(0.0)
            layer._parameters[name].data.uniform_(-0.08,0.08)
        else:
            #nn.init.orthogonal_(layer._parameters[name].data)
            if len(layer._parameters[name].size()) > 1:
                nn.init.kaiming_normal_(layer._parameters[name], mode="fan_out", nonlinearity=nonlinearity)
            
    '''
    if hasattr(layer,"weight"):    
        #nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.uniform_(-0.08,0.08)
        layer.weight.data.mul_(w_scale)
        if hasattr(layer,"bias") and layer.bias is not None:    
            #nn.init.constant_(layer.bias.data, 0)
            layer.bias.data.uniform_(-0.08,0.08)
        
    if hasattr(layer,"weight_ih"):
        #nn.init.orthogonal_(layer.weight_ih.data)
        layer.weight.data.uniform_(-0.08,0.08)
        layer.weight_ih.data.mul_(w_scale)
        if hasattr(layer,"bias_ih"):    
            #nn.init.constant_(layer.bias_ih.data, 0)
            layer.bias.data.uniform_(-0.08,0.08)
        
    if hasattr(layer,"weight_hh"):    
        #nn.init.orthogonal_(layer.weight_hh.data)
        layer.weight.data.uniform_(-0.08,0.08)
        layer.weight_hh.data.mul_(w_scale)
        if hasattr(layer,"bias_hh"):    
            #nn.init.constant_(layer.bias_hh.data, 0)
            layer.bias.data.uniform_(-0.08,0.08)
    '''

    return layer

def layer_init_lstm(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight_ih.data)
    nn.init.orthogonal_(layer.weight_hh.data)
    layer.weight_ih.data.mul_(w_scale)
    layer.weight_hh.data.mul_(w_scale)
    nn.init.constant_(layer.bias_ih.data, 0)
    nn.init.constant_(layer.bias_hh.data, 0)
    return layer

def layer_init_gru(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight_ih.data)
    nn.init.orthogonal_(layer.weight_hh.data)
    layer.weight_ih.data.mul_(w_scale)
    layer.weight_hh.data.mul_(w_scale)
    nn.init.constant_(layer.bias_ih.data, 0)
    nn.init.constant_(layer.bias_hh.data, 0)
    return layer


def is_leaf(node: Dict):
    return any([ not isinstance(node[key], dict) for key in node.keys()])


def recursive_inplace_update(
    in_dict: Dict,
    extra_dict: Union[Dict, torch.Tensor],
    batch_mask_indices: Optional[torch.Tensor]=None,
    preprocess_fn: Optional[Callable] = None):
    '''
    Taking both :param: in_dict, extra_dict as tree structures,
    adds the nodes of extra_dict into in_dict via tree traversal.
    Extra leaf keys are created if and only if the update is over the whole batch, i.e. :param
    batch_mask_indices: is None.
    :param batch_mask_indices: torch.Tensor of shape (batch_size,), containing batch indices that
                        needs recursive inplace update. If None, everything is updated.
    '''
    if in_dict is None: return None
    if is_leaf(extra_dict):
        for leaf_key in extra_dict:
            # In order to make sure that the lack of deepcopy at this point will not endanger
            # the consistency of the data (since we are slicing at some other parts),
            # or, in other words, to make sure that this is yielding a copy rather than
            # a reference, proceed with caution:
            # WARNING: the following makes a referrence of the elements:
            # listvalue = extra_dict[node_key][leaf_key]
            # RATHER, to generate copies that lets gradient flow but do not share
            # the same data space (i.e. modifying one will leave the other intact), make
            # sure to use the clone() method, as list comprehension does not create new tensors.
            listvalue = [value.clone() for value in extra_dict[leaf_key]]
            in_dict[leaf_key] = listvalue
        return 

    for node_key in extra_dict:
        if node_key not in in_dict: in_dict[node_key] = {}
        if not is_leaf(extra_dict[node_key]):
            recursive_inplace_update(
                in_dict=in_dict[node_key], 
                extra_dict=extra_dict[node_key],
                batch_mask_indices=batch_mask_indices,
                preprocess_fn=preprocess_fn,
            )
        else:
            for leaf_key in extra_dict[node_key]:
                # In order to make sure that the lack of deepcopy at this point will not endanger
                # the consistancy of the data (since we are slicing at some other parts),
                # or, in other words, to make sure that this is yielding a copy rather than
                # a reference, proceed with caution:
                # WARNING: the following makes a referrence of the elements:
                # listvalue = extra_dict[node_key][leaf_key]
                # RATHER, to generate copies that lets gradient flow but do not share
                # the same data space (i.e. modifying one will leave the other intact), make
                # sure to use the clone() method, as list comprehension does not create new tensors.
                listvalue = [value.clone() for value in extra_dict[node_key][leaf_key]]
                if batch_mask_indices is None or batch_mask_indices==[]:
                    in_dict[node_key][leaf_key]= listvalue
                else:
                    for vidx in range(len(in_dict[node_key][leaf_key])):
                        v = listvalue[vidx]
                        if leaf_key not in in_dict[node_key]:   continue
                        new_v = v[batch_mask_indices, ...].clone()
                        if preprocess_fn is not None:   new_v = preprocess_fn(new_v)
                        in_dict[node_key][leaf_key][vidx][batch_mask_indices, ...] = new_v

def copy_hdict(in_dict: Dict):
    '''
    Makes a copy of :param in_dict:.
    '''
    if in_dict is None: return None
    
    out_dict = {key: {} for key in in_dict}
    need_reg = False
    if isinstance(in_dict, list):
        out_dict = {'dummy':{}}
        in_dict = {'dummy':in_dict}
        need_reg = True 

    recursive_inplace_update(
        in_dict=out_dict,
        extra_dict=in_dict,
    )

    if need_reg:
        out_dict = out_dict['dummy']

    return out_dict

def extract_subtree(in_dict: Dict,
                    node_id: str):
    '''
    Extracts a copy of subtree whose root is named :param node_id: from :param in_dict:.
    '''
    queue = [in_dict]
    pointer = None

    while len(queue):
        pointer = queue.pop(0)
        if not isinstance(pointer, dict): continue
        for k in pointer.keys():
            if node_id==k:
                return copy_hdict(pointer[k])
            else:
                queue.append(pointer[k])

    return {}


def _extract_from_rnn_states(rnn_states_batched: Dict,
                             batch_idx: Optional[int]=None,
                             map_keys: Optional[List]=None,
                             post_process_fn:Callable=(lambda x:x)): #['hidden', 'cell']):
    '''
    :param map_keys: List of keys we map the operation to.
    '''
    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        # It is possible that an initial rnn states dict has states for actor and critic, separately,
        # but only the actor will be operated during the take_action interface.
        # Here, we allow the critic rnn states to be skipped:
        if rnn_states_batched[recurrent_submodule_name] is None:    continue
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {}
            eff_map_keys = map_keys if map_keys is not None else rnn_states_batched[recurrent_submodule_name].keys()
            for key in eff_map_keys:
                if key in rnn_states_batched[recurrent_submodule_name]:
                    rnn_states[recurrent_submodule_name][key] = []
                    for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                        value = rnn_states_batched[recurrent_submodule_name][key][idx]
                        if batch_idx is not None:
                            value = value[batch_idx,...].unsqueeze(0)
                        rnn_states[recurrent_submodule_name][key].append(post_process_fn(value))
        else:
            rnn_states[recurrent_submodule_name] = _extract_from_rnn_states(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name], 
                batch_idx=batch_idx,
                post_process_fn=post_process_fn
            )
    return rnn_states



