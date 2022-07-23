import Archi
import yaml 

from Archi.modules.utils import copy_hdict 

def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./rl_esbn_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    assert 'RLHead' in model.modules.keys()
    assert 'KeyValueMemory' in model.modules.keys()
    assert 'key_memory' in model.stream_handler.placeholders['inputs']['KeyValueMemory'].keys()
    assert 'value_memory' in model.stream_handler.placeholders['inputs']['KeyValueMemory'].keys()
    assert 'read_key_plus_conf' in model.stream_handler.placeholders['inputs']['KeyValueMemory'].keys()
    assert 'CoreLSTM' in model.modules.keys()
    assert 'CoreLSTM' in model.stream_handler.placeholders['inputs'].keys()
    assert 'hidden' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'cell' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'iteration' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
   

def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./rl_esbn_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 

    inputs_dict = {
        'obs':torch.rand(4,3,64,64),
        'action': torch.randint(0,8,size=(4,1)),
        'rnn_states':{'legal_actions': [torch.rand(4,8)]},
    }

    prediction = model(**inputs_dict)
    
    print("Model's Predictions:")
    for k,v in prediction.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} : {v.shape}")
        elif isinstance(v, dict):
            for k1,v1 in v.items():
                print(f"{k}:{k1} : {type(v1)}")
        else:
            print(f"{k} : {type(v)}")

    output = model.output_stream_dict
    assert output['inputs']['KeyValueMemory']['read_key_plus_conf'][0].max() == 0.0    
    prediction1 = model(**inputs_dict)
    output1 = model.output_stream_dict

    assert 'lstm_output' in output['modules']['CoreLSTM']
    assert 'processed_input' in output['modules']['ToOutputFCN']
    assert 'qa' in output['modules']['RLHead']
    assert 'ent' in output['modules']['RLHead']
    assert 'log_a' in output['modules']['RLHead']
    assert 'processed_input' in output['modules']['Encoder']
    assert 'processed_input' in output['modules']['ToGateFCN']
    assert output['inputs']['KeyValueMemory']['read_key_plus_conf'][0].max() == 0.0    
    assert output1['inputs']['KeyValueMemory']['read_key_plus_conf'][0].max() != 0.0    
    assert len(dict(model.named_parameters())) != 0
    
    print("Model's Parameters:")
    for np, p in model.named_parameters():
        print(np)

if __name__ == '__main__':
    test_model_loading()
    test_model_forward()

