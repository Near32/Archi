from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 


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


        
 
