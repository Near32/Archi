from typing import Dict,List,Optional

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
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Cache,
    DynamicCache,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from Archi.utils import (
    STR2BT,
    BT2STR,
)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


class ArchiTransformerModule(Module):
    def __init__(
        self,
        model_id,
        id='ArchiTransformerModule_0',
        config={
            'quantize':False,
            'bnb_config': {
                'load_in_4bit':False,
                'bnb_4bit_use_double_quant': False,
                'bnb_4bit_quant_type':'nf4',
                'bnb_4bit_compute_dtype': torch.bfloat16,
            },
            'lora_config': {
                'r':8,
                'lora_alpha':32,
                'target_modules':['q_proj','k_proj','v_proj','o_proj'],
                'lora_dropout': 0.05,
                'bias':'none',
                'task_type':'CAUSAL_LM',
            },
            'gradient_checkpointing': True,
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
        super(ArchiTransformerModule, self).__init__(
            id=id,
            type="ArchiTransformerModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.use_cuda = use_cuda
        self.generation_kwargs = self.config['generation_kwargs']
        
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.quantization_config = None
        if self.config['quantize']:
            if isinstance(self.config['bnb_config']['bnb_4bit_compute_dtype'], str):
                dtype_str = self.config['bnb_config']['bnb_4bit_compute_dtype'] 
                dtype_cls = getattr(torch, dtype_str, None)
                if dtype_cls is None:
                    raise NotImplementedError
                self.config['bnb_config']['bnb_4bit_compute_dtype'] = dtype_cls
            self.quantization_config = BitsAndBytesConfig(
                **self.config['bnb_config'],
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            quantization_config=self.quantization_config,
            device_map={"":0} if self.use_cuda else 'auto',
            trust_remote_code=True,
        )
        if not self.use_cuda:   self.model = self.model.cpu()

        if self.config['quantize']:
            self.model = prepare_model_for_kbit_training(self.model)

        if self.config['use_lora']:
            self.lora_config = LoraConfig(**self.config['lora_config'])
            self.model = get_peft_model(self.model, self.lora_config)
        
        if self.config['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
        else:
            self.model.gradient_checkpointing_disable()

        print_trainable_parameters(self.model)

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
                else:
                    orig_padding_side = self.tokenizer.padding_side
                    self.tokenizer.padding_side = 'right'
                    batched_inputs = self.tokenizer(
                        decoded_x,
                        padding=True,
                        #truncation=True,
                        return_tensors='pt',
                    )   
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
        
        return self._forward_options_cache(
            output_dict=output_dict,
            batched_prompts_inputs=batched_prompts_inputs,
            list_batched_options_inputs=list_batched_options_inputs,
        )

    def _forward_options_nocache(
        self,
        output_dict,
        batched_prompts_inputs,
        list_batched_options_inputs,
    ):
        prompts_batch_size = batched_prompts_inputs['input_ids'].shape[0]
        batch_prompts_inputs = batched_prompts_inputs.to(self.model.device)

        '''
        #Forward prompts and retrieve past_key_values:
        cache = transformers.DynamicCache()
        outputs = self.model(
            **batched_prompts_inputs,
            use_cache=True,
            past_key_values=cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        #past_key_values = outputs.past_key_values
        prompts_predicted_logits = outputs.logits
        prompts_lhidden_states = outputs.hidden_states[-1]
        # (batch_size x sequence_length x embed_size_per_head)
        cache = outputs.past_key_values
        if isinstance(cache, Cache):
            past_key_values = cache.to_legacy_cache()
        else:
            past_key_values = cache
        # (batch_size x num_heads x sequence_length x embed_size_per_head)
        '''

        # Compute the different options for each prompt:
        max_option_len = 0
        max_option_batch_size = 0
        list_predicted_logits = []
        list_options_likelihoods = []
        llist_options_likelihoods = []
        list_options_perplexities = []
        llist_options_perplexities = []

        list_lhidden_states = []
        tokenized_predictions = []
        tokenized_option_predictions = []
        for prompt_idx in range(prompts_batch_size):
            prompt_len = batched_prompts_inputs['input_ids'].shape[1]
            batched_options_inputs = list_batched_options_inputs[prompt_idx].to(self.model.device)
            option_batch_size, option_len = batched_options_inputs['input_ids'].shape[0:2]
            max_option_batch_size = max(max_option_batch_size, option_batch_size)
            #max_option_len = max(max_option_len, option_len)
            max_option_len = max(max_option_len, option_len+prompt_len)
            
            '''
            # Select and Repeat pkv to fit to new input batch_size:
            pkv = [
                [
                    pkv_t[prompt_idx:prompt_idx+1, ...].clone().repeat(option_batch_size,1, 1, 1)
                    for pkv_t in pkv_tuple
                ] for pkv_tuple in past_key_values
            ]
            
            pkv = transformers.DynamicCache.from_legacy_cache(pkv)
            # Check that attention and other elements are ok?
            tokenized_option_predictions.append(batched_options_inputs.input_ids)
            tokenized_prediction = torch.cat([
                batched_prompts_inputs.input_ids[prompt_idx:prompt_idx+1].repeat(option_batch_size, 1),
                batched_options_inputs.input_ids],
                dim=-1,
            )
            tokenized_predictions.append(tokenized_prediction)
            option_outputs = self.model(
                input_ids=batched_options_inputs.input_ids,
                # WARNING: providing the attention_mask is not working, but not necessary since we have right-padded the options.
                # right-pads will be masked out in the computation of the likelihood/perplexity below.
                output_hidden_states=True,
                use_cache=True,
                past_key_values=pkv,
                cache_position=torch.arange(prompt_len,prompt_len+option_len),
                return_dict=True,
            )
            
            '''
            # Testing without pkv:
            nc_batched_options_inputs = {}
            for k in batched_options_inputs.keys():
                #nc_batched_options_inputs[k] = torch.cat(
                batched_options_inputs[k] = torch.cat(
                        [batched_prompts_inputs[k][prompt_idx:prompt_idx+1].repeat(option_batch_size, 1), batched_options_inputs[k]], 
                    dim=-1,
                )
            tokenized_option_predictions.append(batched_options_inputs.input_ids)
            tokenized_prediction = batched_options_inputs.input_ids
            tokenized_predictions.append(tokenized_prediction)
            #print(repr(self.tokenizer.decode(batched_options_inputs.input_ids[0])))
            #import ipdb; ipdb.set_trace()
            option_outputs = nc_option_outputs = self.model(
                #**nc_batched_options_inputs,
                **batched_options_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            
            '''
            nc_logits = nc_option_outputs.logits
            same_nc_logits = nc_logits[:, prompt_len:]
            '''

            option_lhidden_states = option_outputs.hidden_states[-1]
            # (option_batch_size x sequence_length x embed_per_head_size)
            '''
            prompt_lhidden_states = prompts_lhidden_states[prompt_idx].repeat(option_batch_size, 1, 1)
            full_lhidden_states = torch.cat([
                prompt_lhidden_states,
                option_lhidden_states],
                dim=1,
            )
            # (option_batch_size x prompt_len + option_len x embed_size)
            '''
            full_lhidden_states = option_lhidden_states
            list_lhidden_states.append(full_lhidden_states)
            
            predicted_logits = option_outputs.logits
            '''
            all_predicted_logits = torch.cat([
                prompts_predicted_logits[prompt_idx:prompt_idx+1].repeat(option_batch_size, 1, 1),
                predicted_logits],
                dim=1,
            )

            '''
            all_predicted_logits = predicted_logits
            list_predicted_logits.append(predicted_logits)
            # batch_size x max_sentence_length x vocab_size 

            predicted_sentences_length = []
            sentences_likelihoods = []
            sentences_perplexities = []
    
            #Compute perplexity:
            pl = predicted_logits
            #(option_batch_size x max_seq_len x vocab_size)
            softmaxed_pl = pl.softmax(dim=-1)
            slhd = softmaxed_pl.gather(dim=-1, index=batched_options_inputs.input_ids.unsqueeze(-1)).squeeze(-1)
            options_true_length = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).long().sum(dim=-1).unsqueeze(-1)
            #(option_batch_size x 1)
            slhd = torch.pow(slhd, 1.0/options_true_length)
            #(option_batch_size x option_len)
            notpadding_mask = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).long()
            slhd = notpadding_mask*slhd+(1-notpadding_mask)*torch.ones_like(slhd)
            sentences_likelihoods = slhd #= slhd.cpu().prod(dim=-1).to(slhd.device)
            #(option_batch_size x option_len )
            sentences_perplexities = 1.0/(slhd+1e-8)
            # (option_batch_size x option_len)
            
            list_options_likelihoods.append(sentences_likelihoods)
            list_options_perplexities.append(sentences_perplexities)
        
            #Compute perplexity with log:
            lpl = all_predicted_logits #predicted_logits
            #(option_batch_size x max_seq_len x vocab_size)
            lsoftmaxed_pl = lpl.log_softmax(dim=-1)
            lslhd = lsoftmaxed_pl.gather(dim=-1, index=batched_options_inputs.input_ids.unsqueeze(-1)).squeeze(-1)
            #lslhd = lsoftmaxed_pl.gather(dim=-1, index=tokenized_prediction.unsqueeze(-1)).squeeze(-1)
            lnotpadding_mask = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).float()
            #lnotpadding_mask = (tokenized_prediction != self.tokenizer.pad_token_id).float()
            #options_true_length = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).long().sum(dim=-1).unsqueeze(-1)
            #(option_batch_size x 1)
            lslhd = lnotpadding_mask * lslhd
            #torch.pow(slhd, 1.0/options_true_length)
            #(option_batch_size x option_len)
            #print('cache option: ', lslhd.shape)
            #print(lslhd)
            lsentences_likelihoods = lslhd.sum(dim=-1) #= slhd.cpu().prod(dim=-1).to(slhd.device)
            #(option_batch_size x option_len )
            lsentences_perplexities = torch.exp(-lsentences_likelihoods / (lnotpadding_mask.sum(dim=-1)+1e-8)) #1.0/(slhd+1e-8)
            # (option_batch_size x option_len)
            
            llist_options_likelihoods.append(lsentences_likelihoods)
            llist_options_perplexities.append(lsentences_perplexities)
        
        # Stack all the hidden_states:
        hidden_states_size = list_lhidden_states[-1].shape[-1]
        slhidden_states = torch.zeros(prompts_batch_size, max_option_batch_size, max_option_len, hidden_states_size)
        for pidx in range(prompts_batch_size):
            opt_lhs = list_lhidden_states[pidx]
            slhidden_states[pidx, :opt_lhs.shape[0], :opt_lhs.shape[1], ...] = opt_lhs
        
        # Stack all predicted_logits:
        spredicted_logits = torch.zeros(prompts_batch_size, max_option_batch_size, max_option_len, predicted_logits.shape[-1])
        soptions_likelihoods = torch.ones(prompts_batch_size, max_option_batch_size, max_option_len)
        soptions_perplexities = torch.ones(prompts_batch_size, max_option_batch_size, max_option_len)
        for pidx in range(prompts_batch_size):
            opt_logits = list_predicted_logits[pidx]
            spredicted_logits[pidx,:opt_logits.shape[0],:opt_logits.shape[1],...] = opt_logits

            opt_lhd = list_options_likelihoods[pidx]
            soptions_likelihoods[pidx,:opt_lhd.shape[0],:opt_lhd.shape[1]] = opt_lhd
            # Regularise when there is less than max_option_batch_size options for this particular prompt:
            soptions_likelihoods[pidx,opt_lhd.shape[0]:] = 0
            
            opt_ppl = list_options_perplexities[pidx]
            soptions_perplexities[pidx,:opt_ppl.shape[0],:opt_ppl.shape[1]] = opt_ppl
            # Regularise when there is less than max_option_batch_size options for this particular prompt:
            soptions_perplexities[pidx,opt_ppl.shape[0]:] = 0

        # Stack all predicted_logits with log:
        lsoptions_likelihoods = (-torch.inf)*torch.ones(prompts_batch_size, max_option_batch_size)
        lsoptions_perplexities = (torch.inf)*torch.ones(prompts_batch_size, max_option_batch_size)
        for pidx in range(prompts_batch_size):
            opt_lhd = llist_options_likelihoods[pidx]
            lsoptions_likelihoods[pidx,:opt_lhd.shape[0]] = opt_lhd
            
            opt_ppl = llist_options_perplexities[pidx]
            lsoptions_perplexities[pidx,:opt_ppl.shape[0]] = opt_ppl
        
        # Option choosing:
        soptions_probs = soptions_likelihoods.prod(dim=-1)
        # (prompt_batch_size x max_option_batch_size
        if False: #TODO debug self.training:
            #option_distribution = nn.Categorical(logits=soptions_likelihoods.prod(dim=-1))
            # (prompt_batch_size x max_option_batch_size)
            #chosen_option = option_distribution.sample()
            chosen_options = torch.multinomial(soptions_probs, num_samples=1) #.reshape((batch_size,))
            # (prompt_batch_size x 1)
        else:
            chosen_options = soptions_probs.argmax(dim=-1).unsqueeze(-1)
            # (prompt_batch_size x 1)
        
        # Option choosing with log:
        lsoptions_probs = lsoptions_likelihoods.softmax(dim=-1)
        #lsoptions_probs = lsoptions_perplexities 
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
            # regularise for max_option_batch_size:
            tokenized_option_predictions = [
                tk if tk.shape[0]==max_option_batch_size else torch.cat([tk, torch.zeros(max_option_batch_size-tk.shape[0], tk.shape[1]).to(tk.device)], dim=0)
                for tk in tokenized_option_predictions
            ]
            # regularise for max_option_len:
            tokenized_option_predictions = torch.cat(
                [
                    tk if tk.shape[1]==max_option_len else torch.cat([tk, torch.zeros(max_option_batch_size, max_option_len-tk.shape[1]).to(tk.device)], dim=-1)
                for tk in tokenized_option_predictions],
                dim=0,
            )
            tokenized_predictions = torch.cat(
                [
                    tk if tk.shape[1]==max_option_len else torch.cat([tk, torch.zeros(max_option_batch_size, max_option_len-tk.shape[1]).to(tk.device)], dim=-1)
                for tk in tokenized_predictions],
                dim=0,
            )
            output_dict.update({
                'legal_choices': legal_choices,
                # The last token's hidden states are repeating the hidden states of the last non-padding tokens:
                'last_token_last_hidden_states': slhidden_states[:,:,-1,...],
                'last_hidden_states': slhidden_states,
                'tokenized_option_prediction': tokenized_option_predictions,
                'tokenized_prediction': tokenized_predictions,
                'lchosen_options': lchosen_options,
                'chosen_options': chosen_options,
                'prediction_logits': spredicted_logits,
                'prediction_likelihoods': soptions_likelihoods,
                'lprediction_probs': lsoptions_probs,
                'lprediction_likelihoods': lsoptions_likelihoods,
                'prediction_perplexities': soptions_perplexities, 
                'lprediction_perplexities': lsoptions_perplexities, 
            })

        return spredicted_logits

    def _forward_options_cache(
        self,
        output_dict,
        batched_prompts_inputs,
        list_batched_options_inputs,
    ):
        prompts_batch_size = batched_prompts_inputs['input_ids'].shape[0]
        batch_prompts_inputs = batched_prompts_inputs.to(self.model.device)

        #Forward prompts and retrieve past_key_values:
        cache = transformers.DynamicCache()
        outputs = self.model(
            **batched_prompts_inputs,
            use_cache=True,
            past_key_values=cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        prompts_predicted_logits = outputs.logits.cpu()
        prompts_lhidden_states = outputs.hidden_states[-1].cpu()
        # (batch_size x sequence_length x embed_size_per_head)
        cache = outputs.past_key_values
        if isinstance(cache, Cache):
            past_key_values = cache.to_legacy_cache()
        else:
            past_key_values = cache
        # (batch_size x num_heads x sequence_length x embed_size_per_head)
        past_key_values = [
            [ pkvt.cpu() for pkvt in pkvts]
            for pkvts in past_key_values
        ]
        '''
        '''

        # Compute the different options for each prompt:
        max_option_len = 0
        max_option_batch_size = 0
        list_predicted_logits = []
        llist_options_likelihoods = []
        llist_options_perplexities = []

        list_lhidden_states = []
        tokenized_predictions = []
        tokenized_option_predictions = []
        for prompt_idx in range(prompts_batch_size):
            prompt_len = batched_prompts_inputs['input_ids'].shape[1]
            batched_options_inputs = list_batched_options_inputs[prompt_idx].to(self.model.device)
            option_batch_size, option_len = batched_options_inputs['input_ids'].shape[0:2]
            max_option_batch_size = max(max_option_batch_size, option_batch_size)
            #max_option_len = max(max_option_len, option_len)
            max_option_len = max(max_option_len, option_len+prompt_len)
            # Select and Repeat pkv to fit to new input batch_size:
            pkv = [
                [
                    pkv_t[prompt_idx:prompt_idx+1, ...].clone().repeat(option_batch_size,1, 1, 1).to(self.model.device)
                    for pkv_t in pkv_tuple
                ] for pkv_tuple in past_key_values
            ]
            
            pkv = transformers.DynamicCache.from_legacy_cache(pkv)
            # Check that attention and other elements are ok?
            tokenized_option_predictions.append(batched_options_inputs.input_ids.cpu())
            tokenized_prediction = torch.cat([
                batched_prompts_inputs.input_ids[prompt_idx:prompt_idx+1].repeat(option_batch_size, 1).cpu(),
                batched_options_inputs.input_ids.cpu()],
                dim=-1,
            )
            tokenized_predictions.append(tokenized_prediction)
            option_outputs = self.model(
                input_ids=batched_options_inputs.input_ids,
                # WARNING: providing the attention_mask is not working, but not necessary since we have right-padded the options.
                # right-pads will be masked out in the computation of the likelihood/perplexity below.
                output_hidden_states=True,
                use_cache=True,
                past_key_values=pkv,
                cache_position=torch.arange(prompt_len,prompt_len+option_len),
                return_dict=True,
            )
            del pkv

            option_lhidden_states = option_outputs.hidden_states[-1].cpu()
            # (option_batch_size x sequence_length x embed_per_head_size)
            prompt_lhidden_states = prompts_lhidden_states[prompt_idx].repeat(option_batch_size, 1, 1).cpu()
            full_lhidden_states = torch.cat([
                prompt_lhidden_states,
                option_lhidden_states],
                dim=1,
            )
            # (option_batch_size x prompt_len + option_len x embed_size)
            list_lhidden_states.append(full_lhidden_states)
            
            predicted_logits = option_outputs.logits.cpu()
            all_predicted_logits = torch.cat([
                prompts_predicted_logits[prompt_idx:prompt_idx+1].repeat(option_batch_size, 1, 1),
                predicted_logits],
                dim=1,
            )

            list_predicted_logits.append(predicted_logits)
            # batch_size x max_sentence_length x vocab_size 

            predicted_sentences_length = []
            sentences_likelihoods = []
            sentences_perplexities = []
    
            #Compute perplexity with log:
            lpl = all_predicted_logits #predicted_logits
            #(option_batch_size x max_seq_len x vocab_size)
            lsoftmaxed_pl = lpl.log_softmax(dim=-1)
            #lslhd = lsoftmaxed_pl.gather(dim=-1, index=batched_options_inputs.input_ids.unsqueeze(-1)).squeeze(-1)
            lslhd = lsoftmaxed_pl.gather(dim=-1, index=tokenized_prediction.unsqueeze(-1)).squeeze(-1)
            #lnotpadding_mask = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).float()
            lnotpadding_mask = (tokenized_prediction != self.tokenizer.pad_token_id).float()
            #options_true_length = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).long().sum(dim=-1).unsqueeze(-1)
            #(option_batch_size x 1)
            lslhd = lnotpadding_mask * lslhd
            #torch.pow(slhd, 1.0/options_true_length)
            #(option_batch_size x option_len)
            #print('cache option: ', lslhd.shape)
            #print(lslhd)
            lsentences_likelihoods = lslhd.sum(dim=-1) #= slhd.cpu().prod(dim=-1).to(slhd.device)
            #(option_batch_size x option_len )
            lsentences_perplexities = torch.exp(-lsentences_likelihoods / (lnotpadding_mask.sum(dim=-1)+1e-8)) #1.0/(slhd+1e-8)
            # (option_batch_size x option_len)
            
            llist_options_likelihoods.append(lsentences_likelihoods)
            llist_options_perplexities.append(lsentences_perplexities)
        
        # Stack all the hidden_states:
        hidden_states_size = list_lhidden_states[-1].shape[-1]
        slhidden_states = torch.zeros(prompts_batch_size, max_option_batch_size, max_option_len, hidden_states_size)
        for pidx in range(prompts_batch_size):
            opt_lhs = list_lhidden_states[pidx]
            slhidden_states[pidx, :opt_lhs.shape[0], :opt_lhs.shape[1], ...] = opt_lhs
        
        # Stack all predicted_logits with log:
        # Stack all predicted_logits:
        spredicted_logits = torch.zeros(prompts_batch_size, max_option_batch_size, max_option_len, predicted_logits.shape[-1])
        lsoptions_likelihoods = (-torch.inf)*torch.ones(prompts_batch_size, max_option_batch_size)
        lsoptions_perplexities = (torch.inf)*torch.ones(prompts_batch_size, max_option_batch_size)
        for pidx in range(prompts_batch_size):
            opt_logits = list_predicted_logits[pidx]
            spredicted_logits[pidx,:opt_logits.shape[0],:opt_logits.shape[1],...] = opt_logits

            opt_lhd = llist_options_likelihoods[pidx]
            lsoptions_likelihoods[pidx,:opt_lhd.shape[0]] = opt_lhd
            
            opt_ppl = llist_options_perplexities[pidx]
            lsoptions_perplexities[pidx,:opt_ppl.shape[0]] = opt_ppl
        
        # Option choosing with log:
        lsoptions_probs = lsoptions_likelihoods.softmax(dim=-1)
        #lsoptions_probs = lsoptions_perplexities 
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
            # regularise for max_option_batch_size:
            tokenized_option_predictions = [
                tk if tk.shape[0]==max_option_batch_size else torch.cat([tk, torch.zeros(max_option_batch_size-tk.shape[0], tk.shape[1]).to(tk.device)], dim=0)
                for tk in tokenized_option_predictions
            ]
            # regularise for max_option_len:
            tokenized_option_predictions = torch.cat(
                [
                    tk if tk.shape[1]==max_option_len else torch.cat([tk, torch.zeros(max_option_batch_size, max_option_len-tk.shape[1]).to(tk.device)], dim=-1)
                for tk in tokenized_option_predictions],
                dim=0,
            )
            tokenized_predictions = torch.cat(
                [
                    tk if tk.shape[1]==max_option_len else torch.cat([tk, torch.zeros(max_option_batch_size, max_option_len-tk.shape[1]).to(tk.device)], dim=-1)
                for tk in tokenized_predictions],
                dim=0,
            )
            output_dict.update({
                'legal_choices': legal_choices,
                # The last token's hidden states are repeating the hidden states of the last non-padding tokens:
                'last_token_last_hidden_states': slhidden_states[:,:,-1,...],
                'last_hidden_states': slhidden_states,
                'tokenized_option_prediction': tokenized_option_predictions,
                'tokenized_prediction': tokenized_predictions,
                'chosen_options': lchosen_options,
                'prediction_logits': spredicted_logits,
                'prediction_probs': lsoptions_probs,
                'prediction_perplexities': lsoptions_perplexities, 
                'prediction_likelihoods': lsoptions_likelihoods,
            })

        return spredicted_logits
 
    
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


