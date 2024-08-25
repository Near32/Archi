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

try:
    import openai
    import tiktoken
except Exception as e:
    print("Please install openai and tiktoken if you want to use OpenAI API.")

class ArchiHFTGIModule(Module):
    def __init__(
        self,
        model_id,
        id='ArchiHFTGIModule_0',
        config={
            'use_grammar': False,
            'prompt_template':'{prompt}',
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
        self.prompt_template = self.config.get('prompt_template', '{prompt}') 
        self.model_id = model_id

        self.openai_model = ('openai' in model_id[:7].lower())

        if self.openai_model:
            self.model_id = self.model_id.split("openai/")[-1]
            self.init_openai()
        else:
            self.init_hf()

    def init_openai(self):
        self.model = openai.OpenAI(api_key=os.environ.get("OPENAI_API_TOKEN",""))
        # TODO: use tiktoken if they geet a padding mechanism ...
        #self.tokenizer = tiktoken.encoding_for_model(self.model_id)
        assert 'gpt' in self.model_id 
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_hf(self):
        huggingface_hub.login(
            token=os.getenv("HF_API_TOKEN", "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"),
        )
        self.model = InferenceClient(
            model=self.model_id,
            token=os.getenv("HF_API_TOKEN", "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"),
        ) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reset(self):
        pass
    
    def tokenize_options_openai(self, prompts, options):
        '''
        Perfom tokenization for prompts and options separately,
        with left padding for prompt and right padding for options,
        so that the options can be processed using caching of the 
        prompt.
        '''
        #TODO: update with tiktoken when they implement a padding mechanism ...

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

        return batched_prompts_inputs, list_batched_options_inputs

    def tokenize_options_hf(self, prompts, options):
        '''
        Perfom tokenization for prompts and options separately,
        with left padding for prompt and right padding for options,
        so that the options can be processed using caching of the 
        prompt.
        '''
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

        return batched_prompts_inputs, list_batched_options_inputs

    def forward(self, x, gt_sentences=None, output_dict=None):
        '''
        If :param gt_sentences: is not `None`,
        then teacher forcing is implemented...
        '''
        split_options = False
        split_questions = False

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
                if '[NBR_QUESTIONS]' in decoded_x[0]:
                    split_questions = True
                    prompts = [dx.split('[NBR_QUESTIONS]')[0] for dx in decoded_x]
                    nbr_questions = [int(dx.split('[NBR_QUESTIONS]')[1].split('[/NBR_QUESTIONS]')[0]) for dx in decoded_x]
                    max_nbr_options = [int(dx.split('[MAX_NBR_OPTIONS]')[1].split('[/MAX_NBR_OPTIONS]')[0]) for dx in decoded_x]
                elif '[OPTION]' in decoded_x[0]:
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
                    # Adding prompt template to prompt:
                    prompts = [self.prompt_template.format(prompt=prompt) for prompt in prompts]

                    if self.openai_model:
                        batched_prompts_inputs, \
                        list_batched_options_inputs =   self.tokenize_options_openai(
                            prompts=prompts,
                            options=options,
                        )
                    else:
                        batched_prompts_inputs, \
                        list_batched_options_inputs =   self.tokenize_options_hf(
                            prompts=prompts,
                            options=options,
                        )
                    
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
        
        if not split_options and not split_questions:
            return self._forward_inference(
                output_dict=output_dict,
                batched_inputs=batched_inputs,
                gt_sentences=gt_sentences,
            )
        
        if self.config['use_grammar']:
            if split_options:
                return self._forward_grammar_options(
                    output_dict=output_dict,
                    prompts=prompts,
                    options=options,
                )

            elif split_questions:
                return self._forward_grammar_questions(
                    output_dict=output_dict,
                    prompts=prompts,
                    nbr_questions=nbr_questions,
                    max_nbr_options=max_nbr_options,
                )
            else:
                raise NotImplementedError

        return self._forward_options(
            output_dict=output_dict,
            prompts=prompts,
            options=options,
            batched_prompts_inputs=batched_prompts_inputs,
            list_batched_options_inputs=list_batched_options_inputs,
        )

    def forward_openai(
        self,
        batched_options_inputs,
        idx,
    ):
        dins = {'model':self.model_id, 'echo':True, 'max_tokens':0, 'logprobs':1}
        dins['prompt'] = self.tokenizer.decode(batched_options_inputs['input_ids'][idx])
        waiting_time = 1 #mins
        response = False
        while not response:
            try:
                option_output = self.model.chat.completions.create(**dins)
                response = True
            except Exception as e:
                response = False
                print(f"ArchiHFTGIModule: exception caught: {e}\n\nWaiting {waiting_time}mins, before retrying.")
                time.sleep(60*int(waiting_time))
                waiting_time *= 1.5
        import ipdb; ipdb.set_trace()
        logprobs = option_output['choices'][0]['logprobs']['token_logprobs']
        return logprobs

    def forward_hf(
        self,
        batched_options_inputs,
        idx,
    ):
        dins = {'details':True, 'return_full_text':True, 'max_new_tokens':1, 'decoder_input_details':True}
        dins['prompt'] = self.tokenizer.decode(batched_options_inputs['input_ids'][idx])
        waiting_time = 1 #mins
        response = False
        while not response:
            try:
                option_output = self.model.text_generation(**dins)
                response = True
            except Exception as e:
                response = False
                print(f"ArchiHFTGIModule: exception caught: {e}\n\nWaiting {waiting_time}mins, before retrying.")
                time.sleep(60*int(waiting_time))
                waiting_time *= 1.5
        logprobs = torch.Tensor([x.logprob for x in option_output.details.prefill if x.logprob is not None])
        return logprobs

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
                if self.openai_model:
                    logprobs = self.forward_openai(batched_options_inputs, idx=idx)
                else:
                    logprobs = self.forward_hf(batched_options_inputs, idx=idx)
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
        # We compute options from perplexities and not from logprobs/likelihoods because the latter
        # gets squashed to 0 when the context length increases, whereas perplexities account for the
        # context length via a normalization, thus preventing the values from being indistinguishable
        # from each other...
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

    def forward_grammar_openai(self, prompt, opts):
        class MultiChoiceAnswer(BaseModel):
            answer_id: int # OpenAI does not allow conint : conint(ge=0, le=len(opts)-1)
        dins = {'model':self.model_id}
        pans = f"Given the context below, answer the following multiple choice question:\n\n{prompt}\n"
        pans += f"\nThe possible choices are detailed below, preceded by their id (from 0 to {len(opts)-1}) :\n"
        for oidx, opt in enumerate(opts):
            pans += f"{oidx}. {opt}\n"
        pans += f"Please use the following schema: {MultiChoiceAnswer.schema()}\n\n"
        # Previously, before prompting: 
        #pans += "What is the digit id of the correct answer?\n\n As an expert, the digit id of the correct answer is "
        # Now, with prompting:
        pans += "What is the digit id of the correct answer?\n" # As an expert, the digit id of the correct answer is "
        pans = self.prompt_template.format(prompt=pans)
        dins['messages'] = [
          {'role': 'system', 'content': 'You are a helpful assistant.'},            
          {'role': 'user', 'content': pans},
        ]
        dins['response_format']=MultiChoiceAnswer
        # TODO: update dictionnary to fit to API :dins.update(self.generation_kwargs)
        responded = False
        waiting_time = 1 #mins
        while not responded:
            try:
                response = self.model.beta.chat.completions.parse(**dins)
                responded = True
            except Exception as e:
                responded = False
                print(f"ArchiHFTGIModule: exception caught: {e}\n\nWaiting {waiting_time} mins, before retrying.")
                time.sleep(60*int(waiting_time))
                waiting_time *= 1.5
        try:
            # Previously:
            #response = yaml.safe_load(response.choices[0].message.parsed)
            # No longer needed, the API returns the object directly. 
            response = response.choices[0].message.parsed
            #print(response)
            response = int(response.answer_id)
        except Exception as e:
            print(f"ArchiHFTGIModule: yaml safe load exception caught: {e}")
            import ipdb; ipdb.set_trace()
            response = 0
        return response 
    
    def forward_grammar_questions_openai(self, prompt, max_options_batch_size, max_questions_batch_size):
        '''
        :param prompt: str formatted with up to :param max_questions_batch_size: closed-form questions, 
          which should be answered with up to :param max_options_batch_size: possible answers.
        :param max_options_batch_size: int
        :param max_questions_batch_size: int
        :return responses: torch.Tensor of shape (max_questions_batch_size, max_options_batch_size)
        '''
        class MultiQuestionMultiChoiceAnswer(BaseModel):
            answer_ids: List[int] # OpenAI does not allow conint : conint(ge=0, le=len(opts)-1)
        dins = {'model':self.model_id}
        pans = f"{prompt}\n\n"
        pans += f"Please use the following schema: {MultiQuestionMultiChoiceAnswer.schema()}\n"
        pans += f"Make sure to concatenate the answers to all (implicit and explicit) questions into your output.\n"
        pans += f"The list of answer_ids must contain {max_questions_batch_size} elements.\n"
        pans = self.prompt_template.format(prompt=pans)
        dins['messages'] = [
          {'role': 'system', 'content': 'You are a helpful assistant.'},            
          {'role': 'user', 'content': pans},
        ]
        dins['response_format']=MultiQuestionMultiChoiceAnswer
        # TODO: update dictionnary to fit to API :dins.update(self.generation_kwargs)
        responded = False
        waiting_time = 1 #mins
        while not responded:
            try:
                response = self.model.beta.chat.completions.parse(**dins)
                responded = True
            except Exception as e:
                responded = False
                print(f"ArchiHFTGIModule: exception caught: {e}\n\nWaiting {waiting_time} mins, before retrying.")
                time.sleep(60*int(waiting_time))
                waiting_time *= 1.5
        print(pans)
        import ipdb; ipdb.set_trace()
        try:
            # Previously:
            #response = yaml.safe_load(response.choices[0].message.parsed)
            # No longer needed, the API returns the object directly. 
            responses_model = response.choices[0].message.parsed
            responses = torch.tensor(responses_model.answer_ids, dtype=torch.long).reshape(max_questions_batch_size, 1)
        except Exception as e:
            print(f"ArchiHFTGIModule: yaml safe load exception caught: {e}")
            import ipdb; ipdb.set_trace()
            responses = torch.zeros((max_questions_batch_size, 1), dtype=torch.long)
        return responses
    
    def forward_grammar_hf(self, prompt, opts):
        class MultiChoiceAnswer(BaseModel):
            answer_id: conint(ge=0, le=len(opts)-1)
        dins = {}
        pans = f"Given the context below, answer the following multiple choice question:\n\n{prompt}\n"
        pans += f"\nThe possible choices are detailed below, preceded by their id (from 0 to {len(opts)-1}) :\n"
        for oidx, opt in enumerate(opts):
            pans += f"{oidx}. {opt}\n"
        pans += f"Please use the following schema: {MultiChoiceAnswer.schema()}\n\n"
        # Previously, before prompting: 
        #pans += "What is the digit id of the correct answer?\n\n As an expert, the digit id of the correct answer is "
        # Now, with prompting:
        pans += "What is the digit id of the correct answer?\n" # As an expert, the digit id of the correct answer is "
        pans = self.prompt_template.format(prompt=pans)
        dins['prompt'] = pans
        dins['details'] = True
        #dins['return_full_text'] = True
        dins['grammar'] = {"type": "json", "value": MultiChoiceAnswer.schema()}
        dins.update(self.generation_kwargs)
        responded = False
        waiting_time = 1 #mins
        while not responded:
            try:
                response = self.model.text_generation(**dins)
                responded = True
            except Exception as e:
                responded = False
                print(f"ArchiHFTGIModule: exception caught: {e}\n\nWaiting {waiting_time} mins, before retrying.")
                time.sleep(60*int(waiting_time))
                waiting_time *= 1.5
        #print(pans)
        try:
            response = yaml.safe_load(response.generated_text)
            #print(response)
            response = int(response['answer_id'])
        except Exception as e:
            print(f"ArchiHFTGIModule: yaml safe load exception caught: {e}")
            import ipdb; ipdb.set_trace()
            response = 0
        return response 
    
    def forward_grammar_questions_hf(self, prompt:str, max_options_batch_size:int, max_questions_batch_size:int):
        '''
        :param prompt: str formatted with up to :param max_questions_batch_size: closed-form questions, 
          which should be answered with up to :param max_options_batch_size: possible answers.
        :param max_options_batch_size: int
        :param max_questions_batch_size: int
        :return responses: torch.Tensor of shape (max_questions_batch_size, max_options_batch_size)
        '''
        class MultiQuestionMultiChoiceAnswer(BaseModel):
            answer_ids: List[conint(ge=0, le=max_options_batch_size-1)]
        dins = {}
        pans = f"{prompt}\n\n"
        pans += f"Please use the following schema: {MultiQuestionMultiChoiceAnswer.schema()}\n\n"
        pans += f"Make sure to concatenate the answers to all (implicit and explicit) questions in to your output!\n\n"
        pans = self.prompt_template.format(prompt=pans)
        dins['prompt'] = pans
        dins['details'] = True
        #dins['return_full_text'] = True
        import ipdb; ipdb.set_trace()
        #TODO: dins['grammar'] = {"type": "json", "value": MultiQuestionMultiChoiceAnswer.schema()}
        dins.update(self.generation_kwargs)
        responded = False
        waiting_time = 1 #mins
        while not responded:
            try:
                response = self.model.text_generation(**dins)
                responded = True
            except Exception as e:
                responded = False
                print(f"ArchiHFTGIModule: exception caught: {e}\n\nWaiting {waiting_time} mins, before retrying.")
                time.sleep(60*int(waiting_time))
                waiting_time *= 1.5
        print(pans)
        import ipdb; ipdb.set_trace()
        try:
            responses_model = yaml.safe_load(response.generated_text)
            responses = torch.tensor(responses_model['answer_ids'], dtype=torch.long).reshape(max_questions_batch_size, 1)
        except Exception as e:
            print(f"ArchiHFTGIModule: yaml safe load exception caught: {e}")
            import ipdb; ipdb.set_trace()
            responses = torch.zeros((max_questions_batch_size, 1), dtype=torch.long)
        return responses 
    
    def _forward_grammar_options(
        self,
        output_dict,
        prompts,
        options,
    ):
        '''
        This function is a wrapper for _forward_grammar_hf or _forward_grammar_openai.
        It will call the appropriate function based on the value of self.openai_model,
        which is set in the constructor, based on whether model_id contains 'openai'.
        It will use Structured Outputs/Grammar/JSON to extract the answer among the 
        options and update the output_dict accordingly.
        :param output_dict: Dict to update with results.
        :param prompts: list of strings
        :param options: list of list of strings
        :return lchosen_options: (prompt_batch_size x 1)
        ''' 
        responses = []
        max_option_batch_size = 0
        for pidx, opts in enumerate(options):
            prompt = prompts[pidx]
            max_option_batch_size = max(max_option_batch_size, len(opts))
            if self.openai_model:
                response = self.forward_grammar_openai(prompt, opts)
            else:
                response = self.forward_grammar_hf(prompt, opts)
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
 
    
    def _forward_grammar_questions(
        self,
        output_dict,
        prompts:List[str],
        nbr_questions:List[int],
        max_nbr_options:List[int],
    ):
        '''
        This function is a wrapper for _forward_grammar_questions_hf or _forward_grammar_questions_openai.
        It will call the appropriate function based on the value of self.openai_model,
        which is set in the constructor, based on whether model_id contains 'openai'.
        It will use Structured Outputs/Grammar/JSON to extract answers among to 
        the :param nbr_questions: questions found in the prompts, among 
        the :param max_nbr_options: options, and update the output_dict accordingly.

        :param output_dict: Dict to update with results.
        :param prompts: list of strings
        :param nbr_questions: list of integers
        :param max_nbr_options: list of integers
        :return lchosen_options: (prompt_batch_size x nbr_questions x 1)
        ''' 
         
        responses = []
        max_option_batch_size = max(max_nbr_options)
        max_question_batch_size = max(nbr_questions)
        for pidx, prompt in enumerate(prompts):
            if self.openai_model:
                response = self.forward_grammar_questions_openai(prompt, max_option_batch_size, max_question_batch_size)
            else:
                response = self.forward_grammar_questions_hf(prompt, max_option_batch_size, max_question_batch_size)
            # (max_question_batch_size x max_option_batch_size) 
            responses.append(response)        
        response = torch.stack(responses, dim=0)
        # (prompt_batch_size x max_question_batch_size x max_option_batch_size)
 
        lsoptions_likelihoods = torch.zeros(len(prompts), max_question_batch_size, max_option_batch_size)
        # (prompt_batch_size x max_question_batch_size x max_option_batch_size)
        lsoptions_likelihoods = lsoptions_likelihoods.scatter_(
            dim=-1, 
            index=response, 
            src=torch.ones_like(response, dtype=lsoptions_likelihoods.dtype),
        )
        # (prompt_batch_size x max_question_batch_size x max_option_batch_size)

        slhidden_states = torch.zeros(len(prompts), max_question_batch_size, max_option_batch_size, self.config.get('hidden_size', 32))
        # (prompt_batch_size x max_question_batch_size x max_option_batch_size x hidden_size)
        lsoptions_perplexities = lsoptions_likelihoods*(-1)
        # (prompt_batch_size x max_question_batch_size x max_option_batch_size)
        
        # Option choosing with log:
        lsoptions_probs = lsoptions_likelihoods#.softmax(dim=-1)
        # (prompt_batch_size x max_question_batch_size x max_option_batch_size
        if False: #TODO debug self.training:
            #option_distribution = nn.Categorical(logits=soptions_likelihoods.prod(dim=-1))
            # (prompt_batch_size x max_question_batch_size x max_option_batch_size)
            #chosen_option = option_distribution.sample()
            lchosen_options = torch.multinomial(lsoptions_probs, num_samples=1) #.reshape((batch_size,))
            # (prompt_batch_size x max_question_batch_size x 1)
        else:
            lchosen_options = lsoptions_probs.argmax(dim=-1).unsqueeze(-1)
            #lchosen_options = lsoptions_probs.argmin(dim=-1).unsqueeze(-1)
            # (prompt_batch_size x max_question_batch_size x 1)
        
        # Legal choices:
        legal_choices = (lsoptions_probs != 0).long()
        # (prompt_batch_size x max_question_batch_size x max_option_batch_size)

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


