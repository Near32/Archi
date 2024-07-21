from typing import Dict, List 

import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 
from Archi.modules.utils import layer_init

from regym.rl_algorithms.networks import NoisyLinear, EPS


class DuelingLayer(nn.Module):
    def __init__(self, input_dim, action_dim, layer_fn=nn.Linear, layer_init_fn=layer_init):
        super(DuelingLayer, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.advantage = layer_fn(self.input_dim, self.action_dim)
        self.value = layer_fn(self.input_dim, 1)

        if layer_init_fn is not None:
            self.apply(layer_init_fn)

    def forward(self, fx):
        v = self.value(fx)
        adv = self.advantage(fx)
        return v.expand_as(adv) + (adv - adv.mean(1, keepdim=True).expand_as(adv))



class RLCategoricalHeadModule(Module):
    def __init__(
        self, 
        state_dim,   
        action_dim,
        noisy=False,
        dueling=False,
        action_logits_from_probs=False,
        id='RLCategoricalHeadModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        layer_init_fn=layer_init,
        use_cuda=False
    ):

        super(RLCategoricalHeadModule, self).__init__(
            id=id,
            type="RLCategoricalHeadModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.greedy = True
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        self.noisy = noisy 
        self.action_logits_from_probs = action_logits_from_probs

        layer_fn = nn.Linear 
        if self.noisy:  layer_fn = NoisyLinear
        if self.dueling:
            self.fc_critic = DuelingLayer(input_dim=self.state_dim, action_dim=self.action_dim, layer_fn=layer_fn)
        else:
            self.fc_critic = layer_fn(self.state_dim, self.action_dim)
            if layer_init_fn is not None:
                self.fc_critic = layer_init_fn(self.fc_critic, 1e0)

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

        self.feature_dim = self.action_dim

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
        qa = self.fc_critic(phi_features)     
        # batch x action_dim
        return qa
    
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

        qa = self.forward(phi_features)
        
        legal_actions = torch.ones_like(qa)
        if 'legal_actions' in input_streams_dict: 
            legal_actions = input_streams_dict['legal_actions']
            if isinstance(legal_actions, list):  legal_actions = legal_actions[0]
        legal_actions = legal_actions.to(qa.device)
        
        # The following accounts for player dimension if VDN:
        legal_qa = (1+qa-qa.min(dim=-1, keepdim=True)[0]) * legal_actions
        
        probs = None
        if 'probs' in input_streams_dict:
            # Assumed to be already softmax-ed of the last dimension!
            probs = input_streams_dict['probs']
            if isinstance(probs, list): probs = probs[0]
            probs = probs.to(qa.device)
            legal_probs = probs * legal_actions
        else:
            probs = F.softmax( qa, dim=-1 )
            legal_probs = F.softmax( legal_qa, dim=-1 )

        log_probs = torch.log(probs+EPS)
        entropy = -torch.sum(probs*log_probs, dim=-1)
        # batch #x 1
        
        legal_log_probs = torch.log(legal_probs+EPS)
        legal_entropy = -torch.sum(legal_probs*legal_log_probs, dim=-1)
        # batch

        action = None
        if 'action' in input_streams_dict:
            action = input_streams_dict['action']
            if isinstance(action, list):    action = action[0]
        
        action_logits = legal_qa
        if self.action_logits_from_probs:   action_logits = legal_probs
        if action is None:
            if self.greedy:
                #action  = legal_qa.max(dim=-1, keepdim=True)[1]
                action  = action_logits.max(dim=-1, keepdim=True)[1]
            else:
                #action = torch.multinomial(legal_qa.softmax(dim=-1), num_samples=1) #.reshape((batch_size,))
                action = torch.multinomial(action_logits.softmax(dim=-1), num_samples=1) #.reshape((batch_size,))
        # batch #x 1
        
        outputs_stream_dict = {
            'a': action,
            'ent': entropy,
            'legal_ent': legal_entropy,
            'qa': qa,
            'log_a': legal_log_probs,
            'unlegal_log_a': log_probs,
        }
        
        return outputs_stream_dict 

