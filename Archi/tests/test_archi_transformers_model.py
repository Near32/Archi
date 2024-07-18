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

if __name__ == '__main__':
    #test_model_loading()
    test_model_forward()

