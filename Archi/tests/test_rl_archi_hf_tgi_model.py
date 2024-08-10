import torch 
import Archi
import yaml 

from Archi.modules.utils import copy_hdict 
from Archi.utils import (
    BT2STR,
    STR2BT,
)


def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./rl_archi_hf_tgi_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    assert 'RLHead' in model.modules.keys()
    assert 'LMModule' in model.modules.keys()
   

def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./rl_archi_hf_tgi_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    prompts = [
        #"[INST]\nAssume x=22. Compute the result of x+32.\n[/INST]\n\n",
        "Assume x=22. What is the result of x+32?",
        "If I assume x=2, what is the result of x+3?",
    ]
    
    options = [[
        'Ans:4.',
        'Ans:54.',
        'Ans:x+32=60.',
        'Ans:x+32=5.',
        ],[
        '4.',
        '5.',
        '6.',
        '7',
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

    batch_size = 2
    action_dim = 4
    use_cuda = True 

    prompt_options_dict = {
        'obs':bt_opt_sentences,
        'action': torch.randint(0,action_dim,size=(batch_size,1)),
        'rnn_states':{
            'legal_actions': [torch.rand(batch_size,action_dim)],
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
        },
    }

    prediction = model(**prompt_options_dict)
    opt_output = model.output_stream_dict
    
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

    assert 'qa' in output['modules']['RLHead']
    assert 'ent' in output['modules']['RLHead']
    assert 'log_a' in output['modules']['RLHead']
    assert len(dict(model.named_parameters())) != 0
    
    '''
    print("Model's Parameters:")
    for np, p in model.named_parameters():
        print(np)
    '''

if __name__ == '__main__':
    #test_model_loading()
    test_model_forward()

