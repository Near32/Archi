from typing import Dict,List,Optional

import torch
import wandb
from ordered_set import OrderedSet

from Archi.modules.module import Module 
from Archi.modules.utils import (
    copy_hdict,
    apply_on_hdict,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
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
        )

        if self.config['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()

        if self.config['quantize']:
            self.model = prepare_model_for_kbit_training(self.model)

        self.lora_config = LoraConfig(**self.config['lora_config'])
        self.model = get_peft_model(self.model, self.lora_config)
        print_trainable_parameters(self.model)

    def reset(self):
        pass

    def forward(self, x, gt_sentences=None, output_dict=None):
        '''
        If :param gt_sentences: is not `None`,
        then teacher forcing is implemented...
        '''
        if gt_sentences is not None:
            gt_sentences = gt_sentences.long().to(x.device)
        
        if isinstance(x, dict):
            # If input is already tokenized?
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
            elif x.dtype==torch.uint8:
                # it is a byteTensor from string:
                decoded_x = BT2STR(x)
                
            batched_inputs = self.tokenizer(
                decoded_x,
                padding=True,
                #truncation=True,
                return_tensors='pt',
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

        batch_size = batched_inputs['input_ids'].shape[0]
        batch_inputs = batched_inputs.to(self.model.device)

        if gt_sentences is None:
            outputs = self.model.generate(
                **batched_inputs,
                **self.generation_kwargs,
                tokenizer=self.tokenizer,
                return_dict_in_generate=True,
                output_hidden_states=True, 
                output_scores=True, 
                output_logits=True, 
                output_attentions=True, 
                use_cache=True, 
            )
        else:
            outputs = self.model(
                **batched_inputs,
                labels=gt_sentences,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        predicted_logits = outputs.logits
        # batch_size x max_sentence_length x vocab_size 
        #TODO: extract with ST-GS maybe?
        predicted_sentences = outputs.sequences #predicted_argmax_logits = outputs.logits.argmax(dim=-1)
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

        EoS_count = 0
        predicted_sentences_length = []
        sentences_likelihoods = []
        sentences_perplexities = []

        pl = torch.stack(predicted_logits, dim=1)
        softmaxed_pl = pl.softmax(dim=-1)
        for b in range(batch_size):
            end_idx = 0
            for idx_t in range(predicted_sentences.shape[1]):
                if predicted_sentences[b,idx_t] == self.tokenizer.eos_token_id: #self.w2idx['EoS']:
                    EoS_count += 1
                    predicted_sentences[b, idx_t+1:] = self.tokenizer.eos_token_id #self.w2idx['EoS']
                    break
                end_idx += 1
            predicted_sentences_length.append(end_idx)
            #Compute perplexity: 
            # CSGPU2 cuda drive error technical debt:
            #slhd = torch.prod(torch.pow(predicted_argmax_logits[b,:end_idx+1].exp(), 1.0/(end_idx+1)))
            # Dealing with misalignement of predicted_sentences and predicted_logits:
            # previously: slhd = torch.pow(predicted_argmax_logits[b,:end_idx+1].exp(), 1.0/(end_idx+1))
            # now:
            slhd = softmaxed_pl[b].gather(dim=-1, index=predicted_sentences[b,batched_inputs.input_ids.shape[1]:].unsqueeze(-1)).squeeze()
            slhd = torch.pow(slhd, 1.0/slhd.shape[0])
            notpadding_mask = (predicted_sentences[b, batched_inputs.input_ids.shape[1]:] != self.tokenizer.pad_token_id).long()
            slhd = notpadding_mask*slhd+(1-notpadding_mask)*torch.ones_like(slhd)
            slhd = slhd.cpu().prod().to(slhd.device)

            # unstable : torch.prod(predicted_argmax_logits[b,:end_idx+1].exp(), keepdim=False)
            #perplexity = torch.pow(1.0/slhd, 1.0/(end_idx+1))
            perplexity = 1.0/(slhd+1e-8)
            sentences_likelihoods.append(slhd)
            sentences_perplexities.append(perplexity)
        
        sentences_likelihoods = torch.stack(sentences_likelihoods, dim=-1)
        sentences_perplexities = torch.stack(sentences_perplexities, dim=-1)
        # batch_size 
        
        try:
            wandb.log({f"{self.id}/EoSRatioPerBatch":float(EoS_count)/batch_size}, commit=False)
        except Exception as e:
            print(f"WARNING: W&B Logging: {e}")
        
        decoded_predictions = [self.tokenizer.decode(s) for s in predicted_sentences] 
        byte_prediction_sentences = STR2BT(decoded_predictions)
        if output_dict is not None:
            output_dict.update({
                'loss': loss_per_item,
                'tokenized_prediction':predicted_sentences, 
                'byte_prediction': byte_prediction_sentences, 
                'prediction_logits':pl, #predicted_logits,
                'prediction_likelihoods':sentences_likelihoods,
                'prediction_perplexities':sentences_perplexities, 
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


