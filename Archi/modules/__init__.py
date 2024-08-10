from .module import Module
from .utils import load_module

from .fully_connected_network_module import FullyConnectedNetworkModule
from .embedding_module import EmbeddingModule
from .convolutional_network_module import ConvolutionalNetworkModule
from .filmed_module import FiLMedModule
from .recurrent_network_module import LSTMModule, GRUModule
from .recurrent_network_module import EmbeddingRNNModule, CaptionRNNModule 
from .recurrent_network_module import OracleTHERModule
from .differentiable_neural_computer_module import DNCModule
from .key_value_memory_module import KeyValueMemoryModule
from .memory_module import MemoryModule
from .read_heads_module import ReadHeadsModule
from .RL_categorical_head_module import RLCategoricalHeadModule
from .RL_categorical_actor_critic_head_module import RLCategoricalActorCriticHeadModule
from .concatenation_operation_module import ConcatenationOperationModule
from .transformers_module import ArchiTransformerModule
from .huggingface_tgi_module import ArchiHFTGIModule

