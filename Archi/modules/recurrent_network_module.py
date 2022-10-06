from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 
from Archi.modules.utils import layer_init

import wandb

    
class LSTMModule(Module):
    def __init__(
        self, 
        state_dim, 
        hidden_units=[256], 
        non_linearities=['None'],
        id='LSTMModule_0',
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
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
            output_stream_ids=output_stream_ids,
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
        
        for k in list(outputs_stream_dict.keys()):
            if k in self.output_stream_ids:
                outputs_stream_dict[self.output_stream_ids[k]] = outputs_stream_dict[k]

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
        output_stream_ids={},
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
            output_stream_ids=output_stream_ids,
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

        for k in outputs_stream_dict.keys():
            if k in self.output_stream_ids:
                outputs_stream_dict[self.output_stream_ids[k]] = outputs_stream_dict[k]

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


class CaptionRNNModule(Module):
    def __init__(
        self,
        max_sentence_length,
        input_dim=64,
        embedding_size=64, 
        hidden_units=256, 
        num_layers=1, 
        vocabulary=None,
        vocab_size=None,
        gate=None, #F.relu, 
        dropout=0.0, 
        rnn_fn="nn.GRU",
        id='CaptionRNNModule_0',
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        super(CaptionRNNModule, self).__init__(
            id=id,
            type="CaptionRNNModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        if vocabulary == 'None':
            vocabulary = 'key ball red green blue purple \
            yellow grey verydark dark neutral light verylight \
            tiny small medium large giant get go fetch go get \
            a fetch a you must fetch a'

        if isinstance(vocabulary, str):
            vocabulary = vocabulary.split(' ')
        
        self.vocabulary = set([w.lower() for w in vocabulary])
        self.vocab_size = vocab_size
        
        # Make padding_idx=0:
        self.vocabulary = ['PAD', 'SoS', 'EoS'] + list(self.vocabulary)
        
        while len(self.vocabulary) < self.vocab_size:
            self.vocabulary.append( f"DUMMY{len(self.vocabulary)}")

        self.w2idx = {}
        self.idx2w = {}
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w

        self.max_sentence_length = max_sentence_length
        self.voc_size = len(self.vocabulary)

        self.embedding_size = embedding_size
        if isinstance(hidden_units, list):  hidden_units=hidden_units[-1]
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        self.gate = gate
        if isinstance(rnn_fn, str):
            rnn_fn = getattr(torch.nn, rnn_fn, None)
            if rnn_fn is None:
                raise NotImplementedError
        
        self.input_dim = input_dim
        self.input_decoder = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.hidden_units)),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
        )
        
        self.rnn_fn = rnn_fn
        self.rnn = rnn_fn(
            input_size=self.embedding_size,
            hidden_size=self.hidden_units, 
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.embedding = nn.Embedding(self.voc_size, self.embedding_size, padding_idx=0)
        self.token_decoder = nn.Sequential(
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.voc_size)),
        )

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, x, gt_sentences=None):
        '''
        If :param gt_sentences: is not `None`,
        then teacher forcing is implemented...
        '''
        if gt_sentences is not None:
            gt_sentences = gt_sentences.long().to(x.device)
        
        batch_size = x.shape[0]
        # POSSIBLE TEMPORAL DIM ...
        x = x.reshape(batch_size, -1)

        # Input Decoding:
        dx = self.input_decoder(x)

        # batch_size x hidden_units
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device) 
        h_0[0] = dx.reshape(batch_size, -1)
        # (num_layers * num_directions, batch, hidden_size)
        
        if self.rnn_fn==nn.LSTM:
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device) 
            decoder_hidden = (h_0,c_0)
        else:
            decoder_hidden = h_0 
        
        decoder_input = self.embedding(torch.LongTensor([[self.w2idx["SoS"]]]).to(x.device))
        # 1 x embedding_size
        decoder_input = decoder_input.reshape(1, 1, -1).repeat(batch_size, 1, 1)
        # batch_size x 1 x embedding_size

        loss_per_item = []

        predicted_sentences = self.w2idx['PAD']*torch.ones(batch_size, self.max_sentence_length, dtype=torch.long).to(x.device)
        for t in range(self.max_sentence_length):
            output, decoder_hidden = self._rnn(decoder_input, h_c=decoder_hidden)
            token_distribution = F.softmax(self.token_decoder(output), dim=-1) 
            idxs_next_token = torch.argmax(token_distribution, dim=1)
            # batch_size x 1
            predicted_sentences[:, t] = idxs_next_token #.unsqueeze(-1)
            
            # Compute loss:
            if gt_sentences is not None:
                mask = (gt_sentences[:, t]!=self.w2idx['PAD']).float().to(x.device)
                # batch_size x 1
                batched_loss = self.criterion(
                    input=token_distribution, 
                    target=gt_sentences[:, t].reshape(batch_size),
                )
                batched_loss *= mask
                loss_per_item.append(batched_loss.unsqueeze(1))
                
            # Preparing next step:
            if gt_sentences is not None:
                # Teacher forcing:
                idxs_next_token = gt_sentences[:, t]
            # batch_size x 1
            decoder_input = self.embedding(idxs_next_token).unsqueeze(1)
            # batch_size x 1 x embedding_size            
        
        for b in range(batch_size):
            end_idx = 0
            for idx_t in range(predicted_sentences.shape[1]):
                if predicted_sentences[b,idx_t] == self.w2idx['EoS']:
                    break
                end_idx += 1
        
        if gt_sentences is not None:
            loss_per_item = torch.cat(loss_per_item, dim=-1).mean(-1)
            # batch_size x max_sentence_length
            accuracies = (predicted_sentences==gt_sentences).float().mean(dim=0)
            mask = (gt_sentences!=self.w2idx['PAD'])
            sentence_accuracies = (predicted_sentences==gt_sentences).float().masked_select(mask).mean()
            output_dict = {
                'prediction':predicted_sentences, 
                'loss_per_item':loss_per_item, 
                'accuracies':accuracies, 
                'sentence_accuracies':sentence_accuracies
            }

            return output_dict

        return predicted_sentences

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

        for key, experiences in input_streams_dict.items():
            if "gt_sentences" in key:   continue

            output_key = f"processed_{key}"
            if key in self.output_stream_ids:
                output_key = self.output_stream_ids[key]
            
            if isinstance(experiences, list):
                assert len(experiences)==1, f"Provided too many input on id:{key}"
                experiences = experiences[0]
            batch_size = experiences.size(0)

            if self.use_cuda:   experiences = experiences.cuda()

            # GT Sentences ?
            gt_key = f"{key}_gt_sentences"
            gt_sentences = input_streams_dict.get(gt_key, None)
            
            output_dict = {}
            if gt_sentences is None:
                output = self.forward(
                    x=experiences,
                    gt_sentences=gt_sentences,
                )
                output_dict['prediction'] = output
            else:
                if isinstance(gt_sentences, list):
                    assert len(gt_sentences) == 1
                    gt_sentences = gt_sentences[0]
                output_dict = self.forward(
                    x=experiences,
                    gt_sentences=gt_sentences,
                )
            
            output_sentences = output_dict['prediction']

            outputs_stream_dict[output_key] = [output_sentences]
            
            for okey, ovalue in output_dict.items():
                outputs_stream_dict[f"inputs:{key}_{okey}"] = [ovalue]
        
        return outputs_stream_dict 

    def _rnn(self, x, h_c):
        batch_size = x.shape[0]
        rnn_outputs, h_c = self.rnn(x, h_c)
        output = rnn_outputs[:,-1,...]
        if self.gate != 'None':
            output = self.gate(output)
        # batch_size x hidden_units 
        return output, h_c
        # batch_size x sequence_length=1 x hidden_units
        # num_layer*num_directions, batch_size, hidden_units
        
    def get_feature_shape(self):
        return self.hidden_units


class EmbeddingRNNModule(Module):
    def __init__(
        self, 
        voc_size, 
        feature_dim=64,
        embedding_size=64, 
        hidden_units=256, 
        num_layers=1, 
        gate=None, #F.relu, 
        dropout=0.0, 
        rnn_fn="nn.GRU",
        padding_idx=0,
        id='EmbeddingRNNModule_0',
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        super(EmbeddingRNNModule, self).__init__(
            id=id,
            type="EmbeddingRNNModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        if isinstance(hidden_units, list):  hidden_units=hidden_units[-1]
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            self.voc_size, 
            self.embedding_size, 
            padding_idx=padding_idx,
        )
        
        self.gate = gate
        if isinstance(rnn_fn, str):
            rnn_fn = getattr(torch.nn, rnn_fn, None)
            if rnn_fn is None:
                raise NotImplementedError

        self.rnn = rnn_fn(
            input_size=self.embedding_size,
            hidden_size=hidden_units, 
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        
        self.feature_dim = feature_dim
        self.decoder_mlp = nn.Linear(hidden_units, self.feature_dim)
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, x):
        batch_size = x.shape[0]
        sentence_length = x.shape[1]
        
        if x.shape[-1] == 1:    x = x.squeeze(-1)
        embeddings = self.embedding(x.long())
        # batch_size x sequence_length x embedding_size

        rnn_outputs, rnn_states = self.rnn(embeddings)
        # batch_size x sequence_length x hidden_units
        # num_layer*num_directions, batch_size, hidden_units
        
        output = rnn_outputs[:,-1,...]
        if self.gate != 'None':
            output = self.gate(output)

        output = self.decoder_mlp(output)

        # batch_size x hidden_units 

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

        for key, experiences in input_streams_dict.items():
            output_key = f"processed_{key}"
            if key in self.output_stream_ids:
                output_key = self.output_stream_ids[key]
            
            if isinstance(experiences, list):
                assert len(experiences)==1, f"Provided too many input on id:{key}"
                experiences = experiences[0]

            if self.use_cuda:   experiences = experiences.cuda()

            output = self.forward(x=experiences)
            outputs_stream_dict[output_key] = [output]
            
        return outputs_stream_dict 


    def get_feature_shape(self):
        return self.feature_dim

