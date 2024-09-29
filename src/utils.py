# src/utils.py
import json
import os
from abc import ABC, abstractmethod
from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from transformers import AutoTokenizer
import torch
from src.model_wrapper import ModelWrapper


class BaseExperiment(ABC):
    def __init__(self, model_id, result_dir, base_prompt="You are a helpful AI assistant. Answer the users queries as best as you can."):
        self.model_id = model_id
        self.result_dir = result_dir
        self.wrapped_model = None
        self.tokenizer = None
        self.base_prompt = base_prompt

    @abstractmethod
    def run(self):
        pass

    @classmethod
    @abstractmethod
    def run_default(cls):
        pass

    def save_results(self, results, filename):
        os.makedirs(self.result_dir, exist_ok=True)
        with open(os.path.join(self.result_dir, filename), 'w') as f:
            json.dump(results, f, indent=2)


    def load_model(self):
        model = AutoAWQForCausalLM.from_quantized(self.model_id, fuse_layers=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.wrapped_model = ModelWrapper(model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wrapped_model.model = self.wrapped_model.model.to(device)
        print(f"Model loaded on {device}")

    def prepare_input(self, prompt):
        chat = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": prompt},
        ]
        tokens = self.tokenizer.apply_chat_template(chat, return_tensors="pt")
        return tokens.to(get_best_device())
    
    def get_terminators(self):
        return [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]