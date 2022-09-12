from typing import Dict, List 

import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 


class ConcatenationOperationModule(Module):
    def __init__(
        self, 
        id='ConcatenationOperationModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
    ):

        super(ConcatenationOperationModule, self).__init__(
            id=id,
            type="ConcatenationOperationModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
    def forward(self, **inputs):
        output = torch.cat([v for k,v in inputs.items()], dim=self.config['dim'])
        return output

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
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

        if isinstance(list(input_streams_dict.values())[0], list):
            nbr_elements = len(list(input_streams_dict.values())[0])
            output_list = []
            for idx in range(nbr_elements):
                inputs = {k:v[idx] for k,v in input_streams_dict.items() if 'input' in k}
                output_list.append( self.forward(**inputs))    
            outputs_stream_dict[f'output'] = output_list
        else:
            inputs = {k:v for k,v in input_streams_dict.items() if 'input' in k}
            outputs_stream_dict[f'output'] = self.forward(**inputs)
        
        for k in input_streams_dict.keys():
            if 'input' not in k:    continue
            if k in self.output_stream_ids:
                outputs_stream_dict[self.output_stream_ids[k]] = outputs_stream_dict['output']

        return outputs_stream_dict 

    def get_feature_shape(self):
        return None



