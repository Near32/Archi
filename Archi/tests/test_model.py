import Archi
import yaml 


def test_module_loading():
    try:
        config = yaml.safe_load(
            open("./model_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi.modules import load_module

    modules = {}
    for mk, m_kwargs in config['modules'].items():
        modules[m_kwargs['id']] = load_module(mk, m_kwargs)
    
    assert 'FCNModule_0' in modules.keys()
    assert 'test_entry' in modules['FCNModule_0'].config.keys()

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

    assert 'processed_input' in output['modules']['ConvNetModule_0']
    assert 'processed_input' in output['modules']['FCNModule_0']


if __name__ == '__main__':
    test_model_forward()
