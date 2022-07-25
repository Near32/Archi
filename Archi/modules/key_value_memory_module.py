from typing import Dict, List 

import torch
from torch import nn, einsum
import torch.nn.functional as F

from Archi.modules.module import Module 
from Archi.modules.fully_connected_network_module import FullyConnectedNetworkModule as FCNM

class KeyValueMemoryModule(Module):
    def __init__(
        self, 
        key_dim=64, 
        value_dim=64, 
        id='KeyValueMemoryModule_0', 
        config=None,
        input_stream_ids=None,
        use_cuda=False
    ):

        super(KeyValueMemoryModule, self).__init__(
            id=id,
            type="KeyValueMemoryModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.confidence_fcn = FCNM(
            state_dim=1,
            hidden_units=[1],
            non_linearities=[None],
            dropout=0.0,
            id='FCNM_confidence',
            config=None,
            input_stream_ids=None,
            use_cuda=use_cuda,
        )

        self.initialize_memories()
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def initialize_memories(self):
        # Constant
        ## Null :
        self.init_key_mem = torch.zeros((1, 1, self.key_dim))
        self.init_value_mem = torch.zeros((1, 1, self.value_dim))

    def get_reset_states(self, cuda=False, repeat=1):
        kh = self.init_key_mem.clone().repeat(repeat, 1, 1)
        if cuda:    kh = kh.cuda()
        vh = self.init_value_mem.clone().repeat(repeat, 1, 1)
        if cuda:    vh = vh.cuda()
        rk_conf = torch.zeros(repeat, self.key_dim+1)
        hd = {
            'key_memory': [kh],
            'value_memory': [vh],
            'read_key_plus_conf': [rk_conf],
        }
        return hd
    
    def forward(
        self, 
        new_key:torch.Tensor, 
        new_value:torch.Tensor,
        key_memory:List[torch.Tensor],
        value_memory:List[torch.Tensor],
        gate:torch.Tensor,
        iteration:List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        output_dict = {}
        
        batch_size = iteration[0].shape[0]
        
        # Masking memory: there cannot be more memory entry than the current iteration count,
        # which can be different on each batch dimension... :
        nbr_memory_items = value_memory.shape[1]
        memory_mask = (torch.arange(nbr_memory_items).unsqueeze(0).repeat(batch_size, 1) <= iteration[0]).long().to(gate.device)
        import ipdb; ipdb.set_trace()
        # TODO: check memory_mask
        not_first_mask = (iteration[0].reshape(-1) > 1) 
        # 1 and not 0, because the CoreLSTM has already updated it...
        # (batch_size, )
        
        new_read_key_plus_conf = torch.zeros(
            (batch_size, self.key_dim+1),
        ).to(gate.device)
        
        if not_first_mask.long().sum() > 0: 
            not_first_indices = torch.masked_select(
                input=not_first_mask.long() * torch.arange(batch_size).to(not_first_mask.device),
                mask=not_first_mask,
            )
            # (1<= dim <=batch_size, )
            # (batch_size, n, value_dim)
            vm = value_memory[0][not_first_indices, ...].to(gate.device)
            import ipdb; ipdb.set_trace()
            vm = vm*memory_mask
            # TODO: check vm for zeroing out on batch dim where iteration
            # (dim, n , value_dim)
            z = new_value[not_first_indices, ...]
            # Similarities:
            sim = einsum('b n d, b d -> b n', vm, z)
            # (dim, n)
            ## Attention weights:
            wv = sim.softmax(dim=-1).unsqueeze(-1)
            # (dim, n, 1)

            ## Confidence weights:
            conf = self.confidence_fcn(
                sim.unsqueeze(-1),
            ).sigmoid()
            # (dim, n, 1)

            ## Compute new key:
            g = gate[not_first_indices, ...]
            # (dim, 1)
            km = key_memory[0][not_first_indices, ...].to(gate.device)
            # (dim, n, key_dim)

            km_plus_conf = torch.cat([km, conf], dim=-1) 
            # (dim, n, key_dim+1)

            new_read_k = (wv * km_plus_conf).sum(dim=1) 
            # (dim, key_dim+1)
            new_read_k = g.sigmoid() * new_read_k
            # (dim, key_dim+1)
            
            new_read_key_plus_conf.scatter_(
                dim=0,
                index=not_first_indices.unsqueeze(-1).repeat(1, self.key_dim+1).to(g.device),
                src=new_read_k,
            )

        # Write values in memory:
        new_key_memory = torch.cat([key_memory[0].to(gate.device), new_key.unsqueeze(1)], dim=1)
        new_value_memory = torch.cat([value_memory[0].to(gate.device), new_value.unsqueeze(1)], dim=1)

        outputs_dict = {
            'key_memory': [new_key_memory],
            'value_memory': [new_value_memory],
            'read_key_plus_conf': [new_read_key_plus_conf],
        }

        return outputs_dict

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object]:
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict:  
        """

        outputs_stream_dict = {}
        
        iteration = input_streams_dict['iteration']

        new_value = input_streams_dict['new_value']
        new_key = input_streams_dict['new_key']
        value_memory = input_streams_dict['value_memory']
        key_memory = input_streams_dict['key_memory']
        
        gate = input_streams_dict['gate']

        output_dict = self.forward(
            iteration=iteration,
            key_memory=key_memory,
            value_memory=value_memory,
            new_value=new_value,
            new_key=new_key,
            gate=gate,
        )

        outputs_stream_dict.update(output_dict)

        # Bookkeeping:
        for k,v in output_dict.items():
            outputs_stream_dict[f"inputs:{self.id}:{k}"] = v

        return outputs_stream_dict

