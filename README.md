# LLaMA Feature Analysis

This project provides a suite of experiments for analyzing feature utilization in the LLaMA 3.1 language model. It includes tools for investigating feature sensitivity, planning ahead capabilities, and benchmarking performance on a subset of the MMLU dataset.

## Project Structure
```
/experiments
feature_sensitivity.py
mmlu_benchmark.py
planning_ahead.py
/results
/src
model_wrapper.py
utils.py
.gitattributes
.gitignore
LICENSE
modified_block.py
run_all_experiments.py
setup.py
```
## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/llama-feature-analysis.git
cd llama-feature-analysis
```
2. Set up a Python environment:

It's recommended to use a virtual environment. You can use conda or venv.

Using conda:
```
conda create -n llama-analysis python=3.8
conda activate llama-analysis
```
Or using venv:
```
python -m venv llama-env
source llama-env/bin/activate  # On Windows use llama-env\Scripts\activate
```
3. Install PyTorch:

If you don't have PyTorch installed or want to ensure you have a CUDA-enabled version, visit https://pytorch.org/get-started/locally/ and follow the instructions for your system.

For example, for CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

4. Install the package and its dependencies:
```
pip install -e .
```
This will automatically install all required dependencies and replace the necessary file in the AutoAWQ package.

## Usage

### Running All Experiments

To run all experiments sequentially:
```
python run_all_experiments.py
```

### Running Individual Experiments

You can run each experiment individually:
```
python experiments/feature_sensitivity.py
python experiments/planning_ahead.py
python experiments/mmlu_benchmark.py
```

## Experiments

1. **Feature Sensitivity**: Investigates how many features in each transformer layer contribute meaningfully to the model's output as well as under which p value the model starts breaking down.

2. **Planning Ahead**: Explores the model's capability to "think ahead" by analyzing output divergence when early token features are modified.

3. **MMLU Benchmark**: Evaluates the model's performance on a subset of the MMLU dataset under different feature pruning conditions.

## Results

Experiment results are saved in the `/results` directory. Each experiment generates its own set of result files, which can include JSON data, visualizations, and analysis reports.

## Customization

You can customize experiment parameters by modifying the `run_default()` method in each experiment file.

## Contributing

Contributions to this project are welcome. Please ensure that your code adheres to the project's coding standards and include appropriate tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the AutoAWQ package for quantized inference.
- MMLU dataset provided by [cais/mmlu](https://huggingface.co/datasets/cais/mmlu).
- Llama 3.1 by [Meta](https://ai.meta.com/blog/meta-llama-3-1/).