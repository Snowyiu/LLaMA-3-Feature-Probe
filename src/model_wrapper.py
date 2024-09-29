import torch
from awq.modules.fused.block import LlamaLikeBlock

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.layers = [module for module in model.modules() if isinstance(module, LlamaLikeBlock)]
        self.reset_stats()

    def reset_stats(self):
        for layer in self.layers:
            layer.activation_stats = {
                'min': [],
                'max': [],
                'avg': [],
                'count': torch.zeros(layer.hidden_size, dtype=torch.long)
            }

    def set_p_value(self, p):
        for layer in self.layers:
            layer.p = p

    def set_modifications(self, modifications):
        for layer, mods in zip(self.layers, modifications):
            layer.set_modifications(mods)

    def reset_passes(self):
        for layer in self.layers:
            layer.passes = 0

    def get_activation_stats(self):
        return [layer.activation_stats for layer in self.layers]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)