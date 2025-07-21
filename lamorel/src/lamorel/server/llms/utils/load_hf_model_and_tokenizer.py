from enum import Enum
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

class ModelTypesEnum(Enum):
    causal = AutoModelForCausalLM
    seq2seq = AutoModelForSeq2SeqLM


def load_hf_model_and_tokenizer(type, path, pretrained):
    print("Loading model {}".format(path))
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # config.vocab_size = len(tokenizer)

    n_layers_key = 'num_hidden_layers'
    if hasattr(config, "attribute_map") and n_layers_key in config.attribute_map:
        n_layers_key = config.attribute_map[n_layers_key]

    n_layers = getattr(config, n_layers_key)
    model_class = ModelTypesEnum[type].value
    
    if pretrained:
        model_method = lambda **kwargs: model_class.from_pretrained(path, **kwargs)
    else:
        model_method = lambda **kwargs: model_class.from_config(config, **kwargs)

    return tokenizer, model_method, n_layers

def load_hf_model_with_embedding(type, path):
    
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    
    model_class = ModelTypesEnum[type].value
    
    pretrained_model = model_class.from_pretrained(path)
    pretrained_embeddings = pretrained_model.get_input_embeddings()
    
    new_embeddings = nn.Embedding(
        num_embeddings=pretrained_embeddings.num_embeddings,
        embedding_dim=pretrained_embeddings.embedding_dim
    )
    
    new_embeddings.weight.data = pretrained_embeddings.weight.data.clone().detach()
    
    model = model_class.from_config(config, trust_remote_code=True)
    
    model.set_input_embeddings(new_embeddings)
    
    model.tie_weights()
    
    del pretrained_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model

