from typing import Dict, List, Tuple

import os
import copy

import torch
import torch.nn as nn

from Archi.utils import StreamHandler
from Archi.modules import Module 

class Model(Module):
    def __init__(
        self,
        module_id: str="Model_0",
        module_type: str="ModelModule",
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
        super(Model, self).__inint__(
            id=module_id,
            type=module_type,
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
        
        # Register Pipelines:
        self.pipelines = self.config['pipelines']

    def forward(self, **kwargs):
        
        for k,v in kwargs.items():
            self.stream_handler.update(f"inputs:{k}", v)
        
        self.stream_handler.reset("logs_dict")
        self.stream_handler.reset("losses_dict")
        self.stream_handler.reset("signals")
        
        self.stream_handler.start_recording_new_entries()

        for pipe_id, pipeline in self.pipelines.items():
            self.stream_handler.serve(pipeline)

        new_streams_dict = self.stream_handler.stop_recording_new_entries()

        return new_streams_dict




