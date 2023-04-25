import Archi
import yaml 


def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./obverter_ether_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    return

def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./obverter_ether_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 

    batch_size = 4
    use_cuda = True 

    inputs_dict = {
        'obs': torch.rand(batch_size,64),
        'rnn_states': {
            'y':torch.rand(4,32),
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
        },
    }

    prediction = model(**inputs_dict)
    output = model.output_stream_dict

    for k,v in prediction.items():
        print(k, v.shape)

    import ipdb; ipdb.set_trace()
    
    for np, p in model.named_parameters():
        print(np)

if __name__ == '__main__':
    #test_model_loading()
    test_model_forward()
