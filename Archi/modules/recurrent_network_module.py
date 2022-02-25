from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 
from Archi.modules.utils import layer_init

    
class LSTMModule(Module):
    def __init__(
        self, 
        state_dim, 
        hidden_units=[256], 
        non_linearities=['None'],
        id='LSTMModule_0',
        config=None,
        input_stream_ids=None,
        use_cuda=False,
    ):
        '''
        
        :param state_dim: dimensions of the input.
        :param hidden_units: list of number of neurons per recurrent hidden layers.
        :param non_linearities: list of activation function to use after each hidden layer, e.g. nn.functional.relu. Default [None].

        '''
        
        #assert 'lstm_input' in input_stream_ids
        if input_stream_ids is not None:
            if 'lstm_hidden' not in input_stream_ids:
                input_stream_ids['lstm_hidden'] = f"inputs:{id}:hidden"
            if 'lstm_cell' not in input_stream_ids:
                input_stream_ids['lstm_cell'] = f"inputs:{id}:cell"
            if 'iteration' not in input_stream_ids:
                input_stream_ids['iteration'] = f"inputs:{id}:iteration"


        super(LSTMModule, self).__init__(
            id=id,
            type='LSTMModule',
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        dims = [state_dim] + hidden_units
        # Consider future cases where we may not want to initialize the LSTMCell(s)
        self.layers = nn.ModuleList(
            [
                layer_init(
                    nn.LSTMCell(
                        dim_in, 
                        dim_out,
                    )
                ) 
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
        )

        self.feature_dim = dims[-1]
        self.non_linearities = non_linearities
        while len(self.non_linearities) < len(self.layers):
            self.non_linearities.append(self.non_linearities[-1])
        for idx, nl in enumerate(self.non_linearities):
            if not isinstance(nl, str):
                raise NotImplementedError
            if nl=='None':
                self.non_linearities[idx] = None
            else:
                nl_cls = getattr(nn, nl, None)
                if nl_cls is None:
                    raise NotImplementedError
                self.non_linearities[idx] = nl_cls()
        

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        hidden_states, cell_states = recurrent_neurons['hidden'], recurrent_neurons['cell']
        iteration = recurrent_neurons.get("iteration", None)
        if iteration is None:
            batch_size = x.shape[0]
            iteration = torch.zeros((batch_size, 1)).to(x.device)
        niteration = [it+1 for it in iteration]

        next_hstates, next_cstates = [], []
        for idx, (layer, hx, cx) in enumerate(zip(self.layers, hidden_states, cell_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
                cx = torch.cat([cx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")

            if self.use_cuda:
                x = x.cuda()
                hx = hx.cuda()
                cx = cx.cuda()

            nhx, ncx = layer(x, (hx, cx))
            next_hstates.append(nhx)
            next_cstates.append(ncx)
            # Consider not applying activation functions on last layer's output
            if self.non_linearities[idx] is not None:
                nhx = self.non_linearities[idx](nhx)
            x = nhx

        return nhx, {'hidden': next_hstates, 'cell': next_cstates, 'iteration': niteration}
    
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
        
        lstm_input = input_streams_dict['lstm_input']
        lstm_hidden = input_streams_dict['lstm_hidden']
        lstm_cell = input_streams_dict['lstm_cell']
        iteration = input_streams_dict['iteration']
        
        lstm_output, state_dict = self.forward((
            lstm_input if not isinstance(lstm_input, list) else lstm_input[0],
            {
                'hidden': lstm_hidden,
                'cell': lstm_cell,
                'iteration': iteration,
            }),
        )
        
        outputs_stream_dict[f'lstm_output'] = lstm_output
        
        outputs_stream_dict[f'lstm_hidden'] = state_dict['hidden']
        outputs_stream_dict[f'lstm_cell'] = state_dict['cell']
        outputs_stream_dict[f'iteration'] = state_dict['iteration']
        
        # Bookkeeping:
        outputs_stream_dict[f'inputs:{self.id}:hidden'] = state_dict['hidden']
        outputs_stream_dict[f'inputs:{self.id}:cell'] = state_dict['cell']
        outputs_stream_dict[f'inputs:{self.id}:iteration'] = state_dict['iteration']
        
        return outputs_stream_dict 

    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states, cell_states = [], []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
            cell_states.append(h)
        iteration = torch.zeros((repeat, 1))
        if cuda:    iteration = iteration.cuda()
        iteration = [iteration]
        return {'hidden': hidden_states, 'cell': cell_states, 'iteration': iteration}

    def get_feature_shape(self):
        return self.feature_dim


class GRUModule(Module):
    def __init__(
        self, 
        state_dim, 
        hidden_units=[256], 
        non_linearities=['None'],
        id='GRUModule_0',
        config=None,
        input_stream_ids=None,
        use_cuda=False,
    ):
        '''
        
        :param state_dim: dimensions of the input.
        :param hidden_units: list of number of neurons per recurrent hidden layers.
        :param non_linearities: list of activation function to use after each hidden layer, e.g. nn.functional.relu. Default [None].

        '''
        
        #assert 'gru_input' in input_stream_ids
        if input_stream_ids is not None:
            if 'gru_hidden' not in input_stream_ids:
                input_stream_ids['gru_hidden'] = f"inputs:{id}:hidden"
            if 'iteration' not in input_stream_ids:
                input_stream_ids['iteration'] = f"inputs:{id}:iteration"


        super(GRUModule, self).__init__(
            id=id,
            type='GRUModule',
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        dims = [state_dim] + hidden_units
        # Consider future cases where we may not want to initialize the LSTMCell(s)
        self.layers = nn.ModuleList(
            [
                layer_init(
                    nn.GRUCell(
                        dim_in, 
                        dim_out,
                    )
                ) 
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
        )

        self.feature_dim = dims[-1]
        self.non_linearities = non_linearities
        while len(self.non_linearities) < len(self.layers):
            self.non_linearities.append(self.non_linearities[-1])
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        hidden_states = recurrent_neurons['hidden']
        
        iteration = recurrent_neurons.get("iteration", None)
        if iteration is None:
            batch_size = x.shape[0]
            iteration = torch.zeros((batch_size, 1)).to(x.device)
        niteration = [it+1 for it in iteration]

        next_hstates = []
        for idx, (layer, hx) in enumerate(zip(self.layers, hidden_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")
            
            if self.use_cuda:
                x = x.cuda()
                hx = hx.cuda()

            nhx = layer(x, hx)
            next_hstates.append(nhx)
            # Consider not applying activation functions on last layer's output
            if self.non_linearities[idx] is not None:
                nhx = self.non_linearities[idx](nhx)
            x = nhx
        return nhx, {'hidden': next_hstatesi, 'iteration': niteration}

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
        
        gru_input = input_streams_dict['gru_input']
        gru_hidden = input_streams_dict['gru_hidden']
        iteration = input_streams_dict['iteration']

        gru_output, state_dict = self.forward((
            gru_input if not isinstance(gru_input, list) else gru_input[0],
            {
                'hidden': gru_hidden,
                'iteration': iteration,
            }),
        )
        
        outputs_stream_dict[f'gru_output'] = gru_output
        
        outputs_stream_dict[f'gru_hidden'] = state_dict['hidden']
        outputs_stream_dict[f'iteration'] = state_dict['iteration']

        # Bookkeeping:
        outputs_stream_dict[f'inputs:{self.id}:hidden'] = state_dict['hidden']
        outputs_stream_dict[f'inputs:{self.id}:iteration'] = state_dict['iteration']

        return outputs_stream_dict 

    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states = []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
        iteration = torch.zeros((repeat, 1))
        if cuda:    iteration = iteration.cuda()
        iteration = [iteration]
        return {'hidden': hidden_states, 'iteration': iteration}

    def get_feature_shape(self):
        return self.feature_dim


