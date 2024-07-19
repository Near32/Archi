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

def test_model_forward_with_cache_and_options():
    try:
        config = yaml.safe_load(
            open("./archi_transformers_model_cache_and_options_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    prompts = [
        '[INST]\nIn one sentence, define the city of Paris, France.\n[/INST]\n',
        '[INST]\nIn one sentence, define the city of New York, USA.\n[/INST]\n',
    ]
    
    options = [[
        'Paris is the capital of France.',
        'Paris is the most populated city of France and its capital.',
        ],[
        'Answer:New York is the capital of the USA.',
        'Answer:New York is the most populated city of the USA and its capital.',
        ]
    ]
    
    sentences = []
    opt_sentences = []
    for pidx, prompt in enumerate(prompts):
        opt_prompt = prompt+'[/PROMPT]'
        opt_sentence = "[OPTION]".join(options[pidx])
        opt_sentences.append(opt_prompt+opt_sentence)
        for opt in options[pidx]:
            sentences.append(prompt+opt)

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
    
    print(opt_sentences)
    print( opt_output['inputs']['LMModule']['inputs_prediction_likelihoods'][0].prod(dim=-1))
    assert opt_output['inputs']['LMModule']['inputs_prediction_likelihoods'][0].shape == torch.Size((2,2,16))

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    #test_model_loading()
    #test_model_forward()
    test_model_forward_with_cache_and_options()

