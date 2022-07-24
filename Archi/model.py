from typing import Dict, List, Tuple

import os
import copy

import torch
import torch.nn as nn

from Archi.utils import StreamHandler
from Archi.modules import Module, load_module 
from Archi.modules.utils import copy_hdict


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

    def _forward(self, pipelines=None, **kwargs):
        if pipelines is None:
            pipelines = self.pipelines

        batch_size = 1
        for k,v in kwargs.items():
            if batch_size == 1\
            and v is not None:
                batch_size = v.shape[0]
            self.stream_handler.update(f"inputs:{k}", v)
        if self.batch_size != batch_size: 
            self.reset_states(batch_size=batch_size)

        self.stream_handler.reset("logs_dict")
        self.stream_handler.reset("losses_dict")
        self.stream_handler.reset("signals")
        
        self.stream_handler.start_recording_new_entries()

        for pipe_id, pipeline in pipelines.items():
            self.stream_handler.serve(pipeline)

        new_streams_dict = self.stream_handler.stop_recording_new_entries()
        self.data_dict = {'inputs':copy_hdict(self.stream_handler.get_data()['inputs'])}

	# Output mapping:
        for k,v in self.config['output_mappings'].items():
            new_streams_dict[f"outputs:{k}"] = self.stream_handler[v]

        return new_streams_dict

    def forward(self, obs, action=None, rnn_states=None, goal=None, pipelines=None):
        assert goal is None, "Deprecated goal-oriented usage ; please use frame/rnn_states."
        if pipelines is None:
            pipelines = self.pipelines

        batch_size = obs.shape[0]
        
        self.output_stream_dict = self._forward(
	    pipelines=pipelines,
            obs=obs,
            action=action,
	    **rnn_states,
	)
        
        entropy = self.output_stream_dict["outputs:ent"]
        qa = self.output_stream_dict["outputs:qa"]
        legal_log_probs = self.output_stream_dict["outputs:log_a"]
        
        prediction = {
            'a': action,
            'ent': entropy,
            'qa': qa,
            'log_a': legal_log_probs,
        }
        
        next_rnn_states = {}
        for k in rnn_states.keys():
            next_rnn_states[k] = self.data_dict["inputs"][k]
        
        prediction.update({
            'rnn_states': rnn_states,
            'next_rnn_states': next_rnn_states
        })

        return prediction
    
    def get_torso(self):
        return partial(self.forward, pipelines={"torso":self.pipelines["torso"]})

    def get_head(self):
        return partial(self.forward, pipelines={"head":self.pipellines["head"]})


def load_model(config: Dict[str, object]) -> Model:
    mcfg = {}
    
    mcfg['output_mappings'] = config.get("output_mappings", {}) 
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


