from enum import Enum

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

class ModelTypesEnum(Enum):
    causal = AutoModelForCausalLM
    seq2seq = AutoModelForSeq2SeqLM


def load_hf_model_and_tokenizer(type, path, pretrained, embed_pretrained):
    print("Loading model {}".format(path))
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # Select class according to type
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)

    n_layers_key = 'num_hidden_layers'
    if hasattr(config, "attribute_map") and n_layers_key in config.attribute_map:
        n_layers_key = config.attribute_map[n_layers_key]

    n_layers = getattr(config, n_layers_key)
    model_class = ModelTypesEnum[type].value
    
    if embed_pretrained:
        print(f"Loading model with pretrained embeddings from {path}.")
        model_method = lambda **kwargs: model_class.from_pretrained(path, load_embedding=True, **kwargs)
        
        
        
    elif pretrained:
        print(f"Loading pretrained model from {path}.")
        model_method = lambda **kwargs: model_class.from_pretrained(path, **kwargs)
    else:
        print(f"Loading model from config {path}.")
        model_method = lambda **kwargs: model_class.from_config(config, **kwargs)

    return tokenizer, model_method, n_layers
