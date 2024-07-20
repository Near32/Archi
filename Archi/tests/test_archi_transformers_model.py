import torch
import Archi
import yaml 
import gc

from Archi.utils import (
    BT2STR,
    STR2BT,
)

def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./archi_transformers_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    assert 'LMModule' in model.stream_handler.placeholders['modules'].keys()
   
    del model
    gc.collect()
    torch.cuda.empty_cache()

def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./archi_transformers_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    sentences = [
        '[INST]\nIn one sentence, define the city of Paris, France.\n[/INST]\n\n',
        '[INST]\nQuote the most famous line of Shakespeare.\n[/INST]\n\n',
    ]
    
    bts = STR2BT(sentences)

    batch_size = 2
    use_cuda = True 

    inputs_dict = {
        'obs':bts,
    }

    prediction = model(**inputs_dict)
    output = model.output_stream_dict
    
    output_sentences = BT2STR(output['inputs']['LMModule']['inputs_byte_prediction'][0])
    print(output_sentences)
    
    assert len(dict(model.named_parameters())) != 0
    
    for np, p in model.named_parameters():
        print(np)

    del model
    gc.collect()
    torch.cuda.empty_cache()

# Calculate relative differences
def relative_difference(a, b):
    return abs(a - b).max() / torch.max(abs(a), abs(b)).max()

def test_model_forward_with_cache_and_options():
    try:
        config = yaml.safe_load(
            open("./archi_transformers_model_cache_and_options_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    '''
    prompts = [
        #"[INST]\nIn one sentence, define the city of Paris, France.\n[/INST]\n",
        "[INST]\nAssume x=22. Compute the result of x+32.\n[/INST]\n\n",
        "[INST]\nIn one sentence, define the city of New York, USA.\n[/INST]\n",
        "[INST]\nIn one sentence, define the city of New York, USA.\n[/INST]\n",
        "[INST]\nAssume x=2. Compute the result of x+3.\n[/INST]\n",
        "If I assume x=2, then x+3=",
    ]
    
    options = [[
        #'Paris is the capital of France.',
        #'Paris is the most populated city of France and its capital.',
        'Ans:4.',
        'Ans:54.',
        'Ans:60.',
        ],[
        'Answer:New York is the capital of the USA.',
        #'Answer:New York is the most populated city of the USA and its capital.',
        'Answer:New York is the most populated city of France.',
        ],[
        'New York is the capital of the USA.',
        #'New York is the most populated city of the USA and its capital.',
        'New York is the most populated city of Hungary.',
        ],[
        '4.',
        '5.',
        '6.',
        #'x+2=7',
        #'x-1=2',
        ],[
        '4.',
        '5.',
        #'60.',
        #'x+2=7',
        #'x-1=2',
        ],
    ]
    '''
    prompts = [
        "[INST]\nAssume x=22. Compute the result of x+32.\n[/INST]\n\n",
        "If I assume x=2, then x+3=",
    ]
    
    options = [[
        'Ans:4.',
        'Ans:54.',
        #'Ans:x+32=60.',
        'Ans:x+32=54.',
        ],[
        '4.',
        '5.',
        '6.',
        #'x+2=7',
        #'x-1=2',
        ],
    ]
    
    sentences = []
    opt_sentences = []
    pidx2sids = {}
    sidx = 0
    for pidx, prompt in enumerate(prompts):
        opt_prompt = prompt+'[/PROMPT]'
        opt_sentence = "[OPTION]".join(options[pidx])
        opt_sentences.append(opt_prompt+opt_sentence)
        pidx2sids[pidx] = []
        for opt in options[pidx]:
            pidx2sids[pidx].append(sidx)
            #sentences.append(prompt+" "+opt)
            sentences.append(prompt+opt)
            sidx+=1

    bt_opt_sentences = STR2BT(opt_sentences)
    bt_sentences = STR2BT(sentences)

    prompt_options_dict = {
        'obs':bt_opt_sentences,
    }

    sentences_dict = {
        'obs':bt_sentences,
    }

    s_prediction = model(**sentences_dict)
    s_output = model.output_stream_dict

    opt_prediction = model(**prompt_options_dict)
    opt_output = model.output_stream_dict
    
    for pidx, opts in enumerate(opt_sentences):
        sids = pidx2sids[pidx]
        for sidx in sids:
            print(sentences[sidx])

        print(pidx, opts)
        #assert (s_output['inputs']['LMModule']['inputs_tokenized_prediction'][0][sids] == opt_output['inputs']['LMModule']['inputs_tokenized_prediction'][0][sids].long()).all()
        #print('Tokenization with and without cache were the same.')
        print('full tokens:\n', s_output['inputs']['LMModule']['inputs_tokenized_prediction'][0][sids])
        print('with cache full tokens:\n',opt_output['inputs']['LMModule']['inputs_tokenized_prediction'][0][sids].long())
        #print('with cache option-only tokens:\n',opt_output['inputs']['LMModule']['inputs_tokenized_option_prediction'][0][sids].long())

        #print( opt_output['inputs']['LMModule']['inputs_prediction_likelihoods'][0])
        #print( opt_output['inputs']['LMModule']['inputs_prediction_likelihoods'][0].prod(dim=-1))
        print('full ppl with log: ', s_output['inputs']['LMModule']['inputs_lprediction_perplexities'][0][sids])
        print('full ppl: ', s_output['inputs']['LMModule']['inputs_prediction_perplexities'][0][sids].prod(dim=-1))
        full_ppl_choice_id = s_output['inputs']['LMModule']['inputs_prediction_perplexities'][0][sids].prod(dim=-1).argmin(dim=-1)
        print('full ppl choice: ', sentences[sids[full_ppl_choice_id]])
        print('option ppl: ', opt_output['inputs']['LMModule']['inputs_prediction_perplexities'][0][pidx].prod(dim=-1))
        print('option ppl with log: ', opt_output['inputs']['LMModule']['inputs_lprediction_perplexities'][0][pidx])
        ppl_full = s_output['inputs']['LMModule']['inputs_lprediction_perplexities'][0][sids].cpu()
        ppl_option = opt_output['inputs']['LMModule']['inputs_lprediction_perplexities'][0][pidx].cpu()
        print(f"Relative differences full logit vs option logt:   {relative_difference(ppl_full, ppl_option):.6f}")
        #print( opt_output['inputs']['LMModule']['inputs_prediction_likelihoods'][0].sum(dim=-1).exp())
        print('option ppl choice: ', options[pidx][opt_output['inputs']['LMModule']['inputs_chosen_options'][0][pidx].item()])
        print('option ppl with log choice: ', options[pidx][opt_output['inputs']['LMModule']['inputs_lchosen_options'][0][pidx].item()])
    assert opt_output['inputs']['LMModule']['inputs_chosen_options'][0].shape[0] == len(prompts)
    assert opt_output['inputs']['LMModule']['inputs_prediction_likelihoods'][0].shape[0] == len(prompts)

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    #test_model_loading()
    #test_model_forward()
    test_model_forward_with_cache_and_options()

