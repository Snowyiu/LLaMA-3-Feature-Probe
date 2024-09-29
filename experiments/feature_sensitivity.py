# experiments/feature_sensitivity.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import BaseExperiment
from transformers import GenerationConfig
import torch

class FeatureSensitivityExperiment(BaseExperiment):
    def __init__(self, model_id, result_dir, p_values, prompts, base_prompt=None):
        super().__init__(model_id, result_dir, base_prompt)
        self.p_values = p_values
        self.prompts = prompts

    def run(self):
        self.load_model()
        results = {}

        for p in self.p_values:
            self.wrapped_model.set_p_value(p)
            results[p] = self.run_prompts()

        self.save_results(results, "feature_sensitivity_results.json")

    def run_prompts(self):
        prompt_results = {}
        for prompt in self.prompts:
            input_ids = self.prepare_input(prompt)
            
            generation_config = GenerationConfig(
                do_sample=False,
                num_beams=1,
                max_new_tokens=1024,
                eos_token_id=self.get_terminators()
            )

            with torch.no_grad():
                output = self.wrapped_model.generate(
                    input_ids,
                    generation_config=generation_config
                )

            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            prompt_results[prompt] = {
                "output": decoded_output,
                "stats": self.wrapped_model.get_activation_stats()
            }
        return prompt_results

    @classmethod
    def run_default(cls):
        model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
        experiment = cls(
            model_id,
            "results/feature_sensitivity",
            p_values=[0.5, 0.7, 0.9, 1.0],
            prompts=["What do you like doing on a sunny afternoon?", 
                     "Explain quantum entanglement.", 
                     "Write a short poem about artificial intelligence.",
                     "What is the capital of France?", 
                     '''Three opaque cups are set on a table top-down next to each other. From left to right, these are cup 1, cup 2, cup 3. 
A coin is put under the left-most cup and the following operations are applied to them:
1. The middle and right cups get swapped.
2. The left and right cups get swapped.
3. The left and middle cups get swapped. 
4. The middle and right cups get swapped.
5. The left cup is appended to the right and the right and middle ones shifted to the left.
In which cup position is the coin after these operations have taken place?
Think through this step by step.
a) left
b) middle
c) right
d) the coin is no longer under any of the cups'''
                     ]
        )
        experiment.run()

if __name__ == "__main__":
    FeatureSensitivityExperiment.run_default()