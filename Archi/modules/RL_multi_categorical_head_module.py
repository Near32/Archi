from typing import Dict, List 

import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 
from Archi.modules.utils import layer_init

from regym.rl_algorithms.networks import NoisyLinear, EPS
from Archi.modules.RL_categorical_head_module import DuelingLayer


class RLMultiCategoricalHeadModule(Module):
    def __init__(
        self, 
        state_dim,   
        action_dims:List[int],
        noisy=False,
        dueling=False,
        action_logits_from_probs=False,
        id='RLMultiCategoricalHeadModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        layer_init_fn=layer_init,
        use_cuda=False
    ):
        '''
        :param action_dims: list of integers, specifying the number of available actions for each action dimension, e.g. [2,3] for 2 dimensions, the first one with 2 actions and the second one with 3.
        '''
        super(RLMultiCategoricalHeadModule, self).__init__(
            id=id,
            type="RLMultiCategoricalHeadModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.greedy = True
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.dueling = dueling
        self.noisy = noisy 
        self.action_logits_from_probs = action_logits_from_probs

        layer_fn = nn.Linear 
        if self.noisy:  layer_fn = NoisyLinear
        
        self.fc_critics = nn.ModuleList()
        self.max_action_dim = max(self.action_dims) 
        for i, action_dim in enumerate(self.action_dims): 
            if self.dueling:
                fc_critic = DuelingLayer(input_dim=self.state_dim, action_dim=action_dim, layer_fn=layer_fn)
            else:
                fc_critic = layer_fn(self.state_dim, action_dim)
                if layer_init_fn is not None:
                    fc_critic = layer_init_fn(fc_critic, 1e0)
            self.fc_critics.append(fc_critic)

        if config is not None \
        and 'mlp_nbr_layers' in config:
            batch_norm = config.get('with_batchnorm', False)   
            self.mlp = []
            for lidx in range(config['mlp_nbr_layers']):
                self.mlp += [layer_init_fn( nn.Linear(self.state_dim, self.state_dim, bias=not batch_norm), 1e0)]
                if batch_norm:    
                    self.mlp += [nn.BatchNorm1d(self.state_dim)]
                self.mlp += [nn.ReLU()]
            self.mlp = nn.Sequential(*self.mlp)

        self.feature_dim = self.state_dim

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def get_feature_shape(self):
        return self.feature_dim

    def reset_noise(self):
        self.apply(reset_noisy_layer)

    def forward(self, phi_features):
        if len(phi_features.shape) > 2:
            bs = phi_features.shape[0]
            phi_features = phi_features.reshape(bs, -1)
        qas = []
        for fc_critic in self.fc_critics:
            qa = fc_critic(phi_features)
            # batch x action_dim
            if qa.shape[-1] < self.max_action_dim:
                qa = torch.cat([qa, torch.zeros(qa.shape[0], self.max_action_dim - qa.shape[-1]).to(qa.device)], dim=-1)
            qas.append(qa)
        qas = torch.stack(qas, dim=1)
        # batch x nbr_action x max_action_dim
        return qas
    
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
        
        phi_features_list = [v[0] if isinstance(v, list) else v for k,v in input_streams_dict.items() if 'input' in k]
        if self.use_cuda:   phi_features_list = [v.cuda() for v in phi_features_list]
        phi_features = torch.cat(phi_features_list, dim=-1)
        
        if self.use_cuda:   phi_features = phi_features.cuda()
	
        if hasattr(self, 'mlp'):
            phi_features = self.mlp(phi_features)

        qas = self.forward(phi_features)
        # batch x nbr_action x max_action_dim

        legal_actions = torch.ones_like(qas)
        if 'legal_actions' in input_streams_dict: 
            legal_actions = input_streams_dict['legal_actions']
            if isinstance(legal_actions, list):  legal_actions = legal_actions[0]
        legal_actions = legal_actions.to(qas.device)
        
        # The following accounts for player dimension if VDN:
        legal_qas = (1+qas-qas.min(dim=-1, keepdim=True)[0]) * legal_actions
        # batch x nbr_action x max_action_dim

        probs = None
        if 'probs' in input_streams_dict:
            # Assumed to be already softmax-ed of the last dimension!
            probs = input_streams_dict['probs']
            if isinstance(probs, list): probs = probs[0]
            probs = probs.to(qas.device)
            # batch x nbr_action x max_action_dim
            legal_probs = probs * legal_actions
        else:
            probs = F.softmax( qas, dim=-1 )
            legal_probs = F.softmax( legal_qas, dim=-1 )
            # batch x nbr_action x max_action_dim

        log_probs = torch.log(probs+EPS)
        entropy = -torch.sum(probs*log_probs, dim=-1)
        # batch x nbr_action #x 1
        
        legal_log_probs = torch.log(legal_probs+EPS)
        legal_entropy = -torch.sum(legal_probs*legal_log_probs, dim=-1)
        # batch x nbr_action

        action = None
        if 'action' in input_streams_dict:
            action = input_streams_dict['action']
            if isinstance(action, list):    action = action[0]
            # None or batch x nbr_action x 1

        action_logits = legal_qas
        if self.action_logits_from_probs:   action_logits = legal_probs
        if action is None:
            if self.greedy:
                #action  = legal_qa.max(dim=-1, keepdim=True)[1]
                action  = action_logits.max(dim=-1, keepdim=True)[1]
            else:
                #action = torch.multinomial(legal_qa.softmax(dim=-1), num_samples=1) #.reshape((batch_size,))
                action = torch.multinomial(action_logits.softmax(dim=-1), num_samples=1) #.reshape((batch_size,))
        # batch x nbr_action #x 1
        
        outputs_stream_dict = {
            'a': action,
            'ent': entropy,
            'legal_ent': legal_entropy,
            'qa': qas,
            'log_a': legal_log_probs,
            'unlegal_log_a': log_probs,
        }
        
        return outputs_stream_dict 

