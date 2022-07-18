from typing import Dict, List, Tuple

import os
import copy

import torch
import torch.nn as nn

from Archi.utils import StreamHandler
from Archi.modules import Module, load_module 


class Model(Module):
    def __init__(
        self,
        module_id: str="Model_0",
        config: Dict[str,object]={},
        input_stream_ids: Dict[str,str]={},
    ):
        """
        Expected keys in :param config::
            - 'modules'     : Dict[str,object]={},
            - 'pipelines'   : Dict[str, List[str]]={},
            - 'load_path'   : str,
            - 'save_path'   : str,
        
        """
        super(Model, self).__init__(
            id=module_id,
            type="ModelModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        assert 'modules' in self.config
        assert 'pipelines' in self.config
        
        self.stream_handler = StreamHandler()
        self.stream_handler.register("logs_dict")
        self.stream_handler.register("losses_dict")
        self.stream_handler.register("signals")
        
        # Register Hyperparameters:
        for k,v in self.config.items():
            self.stream_handler.update(f"config:{k}", v)
        
        # Register Modules:
        for k,m in self.config['modules'].items():
            self.stream_handler.update(f"modules:{m.get_id()}:ref", m)
        
        self.modules = self.config['modules']
        for km, vm in self.modules.items():
            self.add_module(km, vm)

        # Register Pipelines:
        self.pipelines = self.config['pipelines']
        
        # Reset States:
        self.reset_states()

    def reset_states(self, batch_size=1, cuda=False):
        self.batch_size = batch_size
        for k,m in self.config['modules'].items():
            if hasattr(m, 'get_reset_states'):
                reset_dict = m.get_reset_states(repeat=batch_size, cuda=cuda)
                for ks, v in reset_dict.items():
                    self.stream_handler.update(f"inputs:{m.get_id()}:{ks}",v)
        return 

    def forward(self, **kwargs):
        batch_size = 1
        for k,v in kwargs.items():
            batch_size = v.shape[0]
            self.stream_handler.update(f"inputs:{k}", v)
        if self.batch_size != batch_size: 
            self.reset_states(batch_size=batch_size)

        self.stream_handler.reset("logs_dict")
        self.stream_handler.reset("losses_dict")
        self.stream_handler.reset("signals")
        
        self.stream_handler.start_recording_new_entries()

        for pipe_id, pipeline in self.pipelines.items():
            self.stream_handler.serve(pipeline)

        new_streams_dict = self.stream_handler.stop_recording_new_entries()

        return new_streams_dict



def load_model(config: Dict[str, object]) -> Model:
    mcfg = {}
    
    mcfg['pipelines'] = config['pipelines']
    mcfg['modules'] = {}
    for mk, m_kwargs in config['modules'].items():
        if 'id' not in m_kwargs:    m_kwargs['id'] = mk
        mcfg['modules'][m_kwargs['id']] = load_module(m_kwargs.pop('type'), m_kwargs)
    
    model = Model(
        module_id = config['model_id'],
        config=mcfg,
        input_stream_ids=config['input_stream_ids'],
    )

    return model 


