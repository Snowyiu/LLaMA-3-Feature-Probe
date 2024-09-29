from transformers import GenerationConfig
import torch
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import BaseExperiment
class MMLUBenchmarkExperiment(BaseExperiment):
    def __init__(self, model_id, result_dir, p_values, subsets, base_prompt=None):
        super().__init__(model_id, result_dir, base_prompt)
        self.p_values = p_values
        self.subsets = subsets

    def load_mmlu_questions(self):
        all_questions = []
        for subset in self.subsets:
            ds = load_dataset("cais/mmlu", subset, split="test")
            all_questions.extend(ds)
        return all_questions

    def run(self):
        self.load_model()
        questions = self.load_mmlu_questions()
        results = {}

        for p in self.p_values:
            self.wrapped_model.set_p_value(p)
            results[p] = self.run_benchmark(questions)

        self.save_results(results, "mmlu_benchmark_results.json")

    def run_benchmark(self, questions):
        correct = 0
        total = len(questions)

        for question in tqdm(questions, desc=f"Running benchmark with p={self.wrapped_model.layers[0].p}"):
            prompt = f"Question: {question['question']}\nChoose the correct answer from the following options:\nA) {question['choices'][0]}\nB) {question['choices'][1]}\nC) {question['choices'][2]}\nD) {question['choices'][3]}"
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
            
            # Check if the correct answer letter is in the output
            correct_letter = chr(ord('A') + question['answer'])
            if correct_letter in decoded_output:
                correct += 1

        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "stats": self.wrapped_model.get_activation_stats()
        }

    @classmethod
    def run_default(cls):
        model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
        experiment = cls(
            model_id,
            "results/mmlu_benchmark",
            p_values=[1.0, 0.9, 0.7, 0.5, 0.3],
            subsets=[
                "college_physics",
                "high_school_geography",
                "high_school_mathematics",
                "global_facts",
                "machine_learning",
                "logical_fallacies"
            ]
        )
        experiment.run()

if __name__ == "__main__":
    MMLUBenchmarkExperiment.run_default()