from typing import Dict,List,Optional

import os
import time
import torch
import wandb
from ordered_set import OrderedSet

from Archi.modules.module import Module 
from Archi.modules.utils import (
    copy_hdict,
    apply_on_hdict,
)

import transformers
from transformers import (
    AutoTokenizer,
)

from Archi.utils import (
    STR2BT,
    BT2STR,
)


import huggingface_hub
from huggingface_hub import InferenceClient
from pydantic import BaseModel, conint
import yaml


class ArchiHFTGIModule(Module):
    def __init__(
        self,
        model_id,
        id='ArchiHFTGIModule_0',
        config={
            'use_grammar': False,
            'generation_kwargs': {
                'max_length':128,
                'do_sample':True,
                'temperature':0.7, 
                'repetition_penalty':1.1, 
                'stop_strings':['\n'],
                'top_p':0.7, 
                'top_k':50,
            },
        },
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        super(ArchiHFTGIModule, self).__init__(
            id=id,
            type="ArchiHFTGIModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.use_cuda = use_cuda
        self.generation_kwargs = self.config['generation_kwargs']
        
        self.model_id = model_id
        huggingface_hub.login(
            token=os.getenv("HF_API_TOKEN", "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"),
        )
        self.model = InferenceClient(
            model=self.model_id,
            token=os.getenv("HF_API_TOKEN", "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"),
        ) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reset(self):
        pass

    def forward(self, x, gt_sentences=None, output_dict=None):
        '''
        If :param gt_sentences: is not `None`,
        then teacher forcing is implemented...
        '''
        split_options = False

        if gt_sentences is not None:
            gt_sentences = gt_sentences.long().to(x.device)
        
        if isinstance(x, dict):
            # If input is already tokenized? and we will want to backpropagate through it#TODO?
            batched_inputs = x
        elif isinstance(x, torch.Tensor): 
            # or it only contains input_ids (after ST-GS maybe?)
            batch_size = x.shape[0]
            if x.min()==0 and x.max()==1:
                assert len(x.shape)==3 and x.shape[2]==self.model.config.vocab_size
                max_sentence_length = x.shape[1]
                x_ids = torch.arange(
                    self.model.config.vocab_size,
                ).reshape(
                    1,1,-1,
                ).repeat(
                    batch_size, max_sentence_length, 1,
                )
                x_ids = (x_ids*x).sum(dim=-1)
                import ipdb; ipdb.set_trace()
                #Is it list of strings?
                decoded_x = self.tokenizer.decode(x_ids)
            else: #if x.dtype==torch.uint8:
                # it is a byteTensor from string:
                decoded_x = BT2STR(x.to(torch.uint8))
                # Are there options?
                if '[OPTION]' in decoded_x[0]:
                    split_options = True
                    prompts = [dx.split('[/PROMPT]')[0] for dx in decoded_x]
                    options = [dx.split('[/PROMPT]')[1].split('[OPTION]') for dx in decoded_x]
                else:
                    raise NotImplementedError
                    orig_padding_side = self.tokenizer.padding_side
                    self.tokenizer.padding_side = 'right'
                    batched_inputs = self.tokenizer(
                        decoded_x,
                        padding=True,
                        #truncation=True,
                        return_tensors='pt',
                    )   
                    self.tokenizer.padding_side = orig_padding_side

                if not self.config['use_grammar']:
                    orig_padding_side = self.tokenizer.padding_side
                    self.tokenizer.padding_size = 'left'
                    batched_prompts_inputs = self.tokenizer(
                        prompts,
                        padding=True,
                        add_special_tokens=False, #TODO : it is probably necessary but unclear how, unless for space regulations.
                        return_tensors='pt',
                    )
                    self.tokenizer.padding_side = 'right'
                    list_batched_options_inputs = []
                    for pidx, opts in enumerate(options):
                        prompt = prompts[pidx]
                        prompt_len = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.shape[-1]
                        #prompt_len = batched_prompts_inputs.input_ids.shape[-1]
                        popts = [prompt+opt for opt in opts]
                        t_popts = self.tokenizer(
                            popts,
                            padding=True,
                            add_special_tokens=False, #TODO: figure out whether it is necessary or not
                            return_tensors='pt',
                        )
                        # Remove the prompt elements:
                        for k in t_popts:
                            t_popts[k] = t_popts[k][:, prompt_len:]
                        list_batched_options_inputs.append(t_popts)
                    self.tokenizer.padding_side = orig_padding_side
        elif isinstance(x, np.ndarray):
            # Or expecting inputs to be made of strings, as numpy array:
            assert len(x.shape) == 1 and isinstance(x[0], str)
            batched_inputs = self.tokenizer(
                x,
                padding=True,
                #truncation=True,
                return_tensors='pt',
            )
        else:
            raise NotImplementedError
        
        if not split_options:
            return self._forward_inference(
                output_dict=output_dict,
                batched_inputs=batched_inputs,
                gt_sentences=gt_sentences,
            )
        
        if self.config['use_grammar']:
            return self._forward_grammar_options(
                output_dict=output_dict,
                prompts=prompts,
                options=options,
            )

        return self._forward_options(
            output_dict=output_dict,
            prompts=prompts,
            options=options,
            batched_prompts_inputs=batched_prompts_inputs,
            list_batched_options_inputs=list_batched_options_inputs,
        )

    def _forward_options(
        self,
        output_dict,
        prompts,
        options,
        batched_prompts_inputs,
        list_batched_options_inputs,
    ):
        prompts_batch_size = batched_prompts_inputs['input_ids'].shape[0]
        #batch_prompts_inputs = batched_prompts_inputs.to(self.model.device)
        
        # Compute the different options for each prompt:
        max_option_len = 0
        max_option_batch_size = 0
        llist_options_likelihoods = []
        llist_options_perplexities = []

        list_lhidden_states = []
        tokenized_predictions = []
        tokenized_option_predictions = []
        for prompt_idx in range(prompts_batch_size):
            prompt_len = batched_prompts_inputs['input_ids'].shape[1]

            prompt = self.tokenizer.decode(batched_prompts_inputs['input_ids'][prompt_idx])
            opts = [self.tokenizer.decode(x) for x in list_batched_options_inputs[prompt_idx].input_ids] 

            batched_options_inputs = list_batched_options_inputs[prompt_idx]#.to(self.model.device)
            option_batch_size, option_len = batched_options_inputs['input_ids'].shape[0:2]
            max_option_batch_size = max(max_option_batch_size, option_batch_size)
            #max_option_len = max(max_option_len, option_len)
            max_option_len = max(max_option_len, option_len+prompt_len)
            
            # Testing without pkv:
            nc_batched_options_inputs = {}
            for k in batched_options_inputs.keys():
                #nc_batched_options_inputs[k] = torch.cat(
                batched_options_inputs[k] = torch.cat(
                        [batched_prompts_inputs[k][prompt_idx:prompt_idx+1].repeat(option_batch_size, 1), batched_options_inputs[k]], 
                    dim=-1,
                )
            
            option_outputs_probs = []
            option_outputs_perplexities = []
            for idx in range(option_batch_size):
                dins = {'details':True, 'return_full_text':True, 'max_new_tokens':1, 'decoder_input_details':True}
                dins['prompt'] = self.tokenizer.decode(batched_options_inputs['input_ids'][idx])
                option_output = self.model.text_generation(**dins)
                logprobs = torch.Tensor([x.logprob for x in option_output.details.prefill if x.logprob is not None])
                lsentences_likelihoods = logprobs.sum().exp().item()
                option_outputs_probs.append(lsentences_likelihoods)
                perplexity = torch.exp(-torch.mean(logprobs)).item()
                option_outputs_perplexities.append(perplexity)
            
            lsentences_likelihoods = torch.tensor(option_outputs_probs)
            lsentences_perplexities = torch.tensor(option_outputs_perplexities)
            llist_options_likelihoods.append(lsentences_likelihoods)
            llist_options_perplexities.append(lsentences_perplexities)
        
        # Stack all predicted_logits with log:
        lsoptions_likelihoods = (-torch.inf)*torch.ones(prompts_batch_size, max_option_batch_size)
        lsoptions_perplexities = (torch.inf)*torch.ones(prompts_batch_size, max_option_batch_size)
        for pidx in range(prompts_batch_size):
            opt_lhd = llist_options_likelihoods[pidx]
            lsoptions_likelihoods[pidx,:opt_lhd.shape[0]] = opt_lhd
            
            opt_ppl = llist_options_perplexities[pidx]
            lsoptions_perplexities[pidx,:opt_ppl.shape[0]] = opt_ppl
        
        '''
        lsoptions_likelihoods = torch.zeros(len(prompts), max_option_batch_size)
        # (prompt_batch_size x max_option_batch_size)
        lsoptions_likelihoods[range(len(prompts)), responses] = 1
        # (prompt_batch_size x max_option_batch_size)
        '''

        slhidden_states = torch.zeros(len(prompts), max_option_batch_size, self.config.get('hidden_size', 32))
        # (prompt_batch_size x max_option_batch_size x hidden_size)
        
        # Option choosing with log:
        lsoptions_probs = (lsoptions_perplexities*(-1)).softmax(dim=-1) #options_likelihoods#.softmax(dim=-1)
        # (prompt_batch_size x max_option_batch_size
        if False: #TODO debug self.training:
            #option_distribution = nn.Categorical(logits=soptions_likelihoods.prod(dim=-1))
            # (prompt_batch_size x max_option_batch_size)
            #chosen_option = option_distribution.sample()
            lchosen_options = torch.multinomial(lsoptions_probs, num_samples=1) #.reshape((batch_size,))
            # (prompt_batch_size x 1)
        else:
            lchosen_options = lsoptions_probs.argmax(dim=-1).unsqueeze(-1)
            #lchosen_options = lsoptions_probs.argmin(dim=-1).unsqueeze(-1)
            # (prompt_batch_size x 1)
        
        # Legal choices:
        legal_choices = torch.ones_like(lsoptions_probs).long() #(lsoptions_probs != 0).long()
        # (prompt_batch_size x max_option_batch_size)

        if output_dict is not None:
            output_dict['legal_choices'] = legal_choices
            # The last token's hidden states are repeating the hidden states of the last non-padding tokens:
            output_dict['last_token_last_hidden_states'] = slhidden_states#[:,:,-1,...]
            output_dict['chosen_options'] = lchosen_options
            output_dict['prediction_probs'] = lsoptions_probs
            output_dict['prediction_perplexities'] = lsoptions_perplexities 
            output_dict['prediction_likelihoods'] = lsoptions_likelihoods

        return lchosen_options #spredicted_logits

    def _forward_grammar_options(
        self,
        output_dict,
        prompts,
        options,
    ):
        
        responses = []
        max_option_batch_size = 0
        for pidx, opts in enumerate(options):
            max_option_batch_size = max(max_option_batch_size, len(opts))
            prompt = prompts[pidx]
            class MultiChoiceAnswer(BaseModel):
                answer_id: conint(ge=0, le=len(opts)-1)
            dins = {}
            pans = f"Given the context below, answer the following multiple choice question:\n\n{prompt}\n"
            pans += f"\nThe possible choices are detailed below, preceded by their id (from 0 to {len(opts)-1}) :\n"
            for oidx, opt in enumerate(opts):
                pans += f"{oidx}. {opt}\n"
            pans += f"Please use the following schema: {MultiChoiceAnswer.schema()}\n\n"
            pans += "What is the digit id of the correct answer?\n\n As an expert, the digit id of the correct answer is "
            dins['prompt'] = pans
            dins['details'] = True
            #dins['return_full_text'] = True
            dins['grammar'] = {"type": "json", "value": MultiChoiceAnswer.schema()}
            dins.update(self.generation_kwargs)
            response = False
            if not response:
                try:
                    response = self.model.text_generation(**dins)
                except Exception as e:
                    response = False
                    import ipdb; ipdb.set_trace()
                    print(f"ArchiHFTGIModule: exception caught: {e}")
                    time.sleep(5)
            #print(pans)
            #import ipdb; ipdb.set_trace()
            try:
                response = yaml.safe_load(response.generated_text)
                #print(response)
                response = int(response['answer_id'])
            except Exception as e:
                print(f"ArchiHFTGIModule: yaml safe load exception caught: {e}")
                import ipdb; ipdb.set_trace()
                response = 0
            responses.append(response)        
        
        lsoptions_likelihoods = torch.zeros(len(prompts), max_option_batch_size)
        # (prompt_batch_size x max_option_batch_size)
        lsoptions_likelihoods[range(len(prompts)), responses] = 1
        # (prompt_batch_size x max_option_batch_size)

        slhidden_states = torch.zeros(len(prompts), max_option_batch_size, self.config.get('hidden_size', 32))
        # (prompt_batch_size x max_option_batch_size x hidden_size)
        lsoptions_perplexities = lsoptions_likelihoods*(-1)
        # (prompt_batch_size x max_option_batch_size)
        
        # Option choosing with log:
        lsoptions_probs = lsoptions_likelihoods#.softmax(dim=-1)
        # (prompt_batch_size x max_option_batch_size
        if False: #TODO debug self.training:
            #option_distribution = nn.Categorical(logits=soptions_likelihoods.prod(dim=-1))
            # (prompt_batch_size x max_option_batch_size)
            #chosen_option = option_distribution.sample()
            lchosen_options = torch.multinomial(lsoptions_probs, num_samples=1) #.reshape((batch_size,))
            # (prompt_batch_size x 1)
        else:
            lchosen_options = lsoptions_probs.argmax(dim=-1).unsqueeze(-1)
            #lchosen_options = lsoptions_probs.argmin(dim=-1).unsqueeze(-1)
            # (prompt_batch_size x 1)
        
        # Legal choices:
        legal_choices = (lsoptions_probs != 0).long()
        # (prompt_batch_size x max_option_batch_size)

        if output_dict is not None:
            output_dict['legal_choices'] = legal_choices
            # The last token's hidden states are repeating the hidden states of the last non-padding tokens:
            output_dict['last_token_last_hidden_states'] = slhidden_states#[:,:,-1,...]
            output_dict['chosen_options'] = lchosen_options
            output_dict['prediction_probs'] = lsoptions_probs
            output_dict['prediction_perplexities'] = lsoptions_perplexities 
            output_dict['prediction_likelihoods'] = lsoptions_likelihoods

        return lchosen_options #spredicted_logits
 
    
    def _forward_inference(
        self,
        output_dict,
        batched_inputs,
        gt_sentences,
    ):
        batch_size = batched_inputs['input_ids'].shape[0]
        batch_inputs = batched_inputs.to(self.model.device)

        if gt_sentences is None:
            '''
            outputs = self.model.generate(
                **batched_inputs,
                **self.generation_kwargs,
                tokenizer=self.tokenizer,
                return_dict_in_generate=True,
                output_hidden_states=True, 
                output_scores=True, 
                output_logits=True, 
                output_attentions=True, 
                use_cache=False,#True, 
            )
            '''
            outputs = self.model(
                #**batched_inputs,
                input_ids=batched_inputs.input_ids,
                #**self.generation_kwargs,
                #tokenizer=self.tokenizer,
                return_dict=True,
                output_hidden_states=True, 
                #output_scores=True, 
                #output_logits=True, 
                #output_attentions=True, 
                use_cache=False,#True, 
            )
        else:
            raise NotImplementedError
            # TODO
            outputs = self.model(
                **batched_inputs,
                labels=gt_sentences,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        slhidden_states = outputs.hidden_states[-1]
        # (batch_size x sequence_length x embed_size_per_head)
        predicted_logits = outputs.logits
        # batch_size x max_sentence_length x vocab_size 
        #TODO: extract with ST-GS maybe?
        predicted_sentences = batched_inputs.input_ids #outputs.sequences #predicted_argmax_logits = outputs.logits.argmax(dim=-1)
        # batch_size x max_sentence_length - with padding... 

        # Compute loss:
        if gt_sentences is not None:
            # Shift so that tokens < n predict n
            shift_logits = predicted_logits[..., :-1, :].contiguous()
            shift_labels = gt_sentences[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss_per_item = loss_fct(shift_logits, shift_labels)
            # (batch_size x max_sentence_length )
        else:
            loss_per_item = torch.zeros(batch_size).to(self.model.device)

        #Compute perplexity:
        pl = predicted_logits #torch.stack(predicted_logits, dim=1)
        #(batch_size x max_seq_len x vocab_size)
        softmaxed_pl = pl.softmax(dim=-1)
        slhd = softmaxed_pl.gather(dim=-1, index=predicted_sentences.unsqueeze(-1)).squeeze(-1)
        true_length = (predicted_sentences != self.tokenizer.pad_token_id).long().sum(dim=-1).unsqueeze(-1)
        #(batch_size x 1)
        slhd = torch.pow(slhd, 1.0/true_length)
        #(batch_size x max_seq_len)
        notpadding_mask = (predicted_sentences != self.tokenizer.pad_token_id).long()
        slhd = notpadding_mask*slhd+(1-notpadding_mask)*torch.ones_like(slhd)
        sentences_likelihoods = slhd #= slhd.cpu().prod(dim=-1).to(slhd.device)
        #(batch_size x max_seq_len )
        sentences_perplexities = 1.0/(slhd+1e-8)
        # (batch_size x max_seq_len)
        
        #Compute perplexity with log:
        lpl = pl #predicted_logits
        #(option_batch_size x max_seq_len x vocab_size)
        lsoftmaxed_pl = lpl.log_softmax(dim=-1)
        lslhd = lsoftmaxed_pl.gather(dim=-1, index=predicted_sentences.unsqueeze(-1)).squeeze(-1)
        lnotpadding_mask = (predicted_sentences != self.tokenizer.pad_token_id).float()
        #options_true_length = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).long().sum(dim=-1).unsqueeze(-1)
        #(option_batch_size x 1)
        lslhd = lnotpadding_mask * lslhd
        #torch.pow(slhd, 1.0/options_true_length)
        #(option_batch_size x option_len)
        print(lslhd.shape)
        print(lslhd)
        lsentences_likelihoods = lslhd.sum(dim=-1) #= slhd.cpu().prod(dim=-1).to(slhd.device)
        #(option_batch_size x option_len )
        lsentences_perplexities = torch.exp(-lsentences_likelihoods / (lnotpadding_mask.sum(dim=-1)+1e-8)) #1.0/(slhd+1e-8)
        # (option_batch_size x option_len)
        
        # Legal choices:
        # TODO: adapt from options, but unlikely to be feasible
        #legal_choices = (lsoptions_probs != 0).long()
        # (prompt_batch_size x max_option_batch_size)

        decoded_predictions = [self.tokenizer.decode(s) for s in predicted_sentences] 
        byte_prediction_sentences = STR2BT(decoded_predictions)
        if output_dict is not None:
            output_dict.update({
                'loss': loss_per_item,
                # The last token's hidden states are repeating the hidden states of the last non-padding tokens:
                'last_token_last_hidden_states': slhidden_states[:,-1,...],
                'last_hidden_states': slhidden_states,
                'tokenized_prediction':predicted_sentences, 
                'byte_prediction': byte_prediction_sentences, 
                'prediction_logits':pl, #predicted_logits,
                'prediction_likelihoods':sentences_likelihoods,
                'prediction_perplexities':sentences_perplexities, 
                'lprediction_perplexities':lsentences_perplexities, 
            })

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
            #TODO : What are the experiences like?
            batch_size = experiences.size(0)
            
            if len(experiences.shape)>2 \
            and experiences.shape[-1] != experiences.shape[-2]:
                # if it is not a feature map but it has an extra dimension:
                experiences = experiences.reshape((batch_size, -1))

            # GT Sentences ?
            gt_key = f"{key}_gt_sentences"
            gt_sentences = input_streams_dict.get(gt_key, None)
            
            output_dict = {}
            if gt_sentences is None:
                output = self.forward(
                    x=experiences,
                    gt_sentences=gt_sentences,
                    output_dict=output_dict,
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
                outputs_stream_dict[f"inputs:{self.id}:{key}_{okey}"] = [ovalue]
        
        return outputs_stream_dict 

    def get_feature_shape(self):
        return self.model.config.vocab_size


