from transformers import LlamaForCausalLM
import torch



class LlamaForCausalLMTeacher(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config.num_hidden_layers = 3
        super().__init__(config)

    def reset_para(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                continue
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

