import Archi
import yaml 


def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./model_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    assert 'FCNModule_0' in model.modules.keys()
    assert 'test_entry' in model.modules['FCNModule_0'].config.keys()
    assert 'CoreLSTM' in model.modules.keys()
    assert 'CoreLSTM' in model.stream_handler.placeholders['inputs'].keys()
    assert 'hidden' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'cell' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'SecondaryDNC' in model.modules.keys()
    assert 'SecondaryDNC' in model.stream_handler.placeholders['inputs'].keys()
    assert 'dnc' in model.stream_handler.placeholders['inputs']['SecondaryDNC'].keys()
    assert 'dnc_body' in model.stream_handler.placeholders['inputs']['SecondaryDNC']['dnc'].keys()
    assert 'dnc_controller' in model.stream_handler.placeholders['inputs']['SecondaryDNC']['dnc'].keys()
    assert 'dnc_memory' in model.stream_handler.placeholders['inputs']['SecondaryDNC']['dnc'].keys()
    
def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./model_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 

    inputs_dict = {
        'x':torch.rand(4,3,64,64),
        'y':torch.rand(4,32),
    }

    output = model(**inputs_dict)
    output1 = model(**inputs_dict)

    assert 'processed_input' in output['modules']['ConvNetModule_0']
    assert 'processed_input' in output['modules']['FCNModule_0']
    assert 'lstm_output' in output['modules']['CoreLSTM']
    assert 'dnc_output' in output['modules']['SecondaryDNC']
    assert output1['inputs']['SecondaryDNC']['dnc']['dnc_body']['prev_read_vec'][0].max() != 0.0    
    assert len(dict(model.named_parameters())) != 0

if __name__ == '__main__':
    test_model_loading()
    test_model_forward()

