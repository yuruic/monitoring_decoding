# Monitoring Decoding

A repository for implementing and evaluating monitoring-based decoding strategies for mitigating hallucination in LLM generation.

## Overview

This repository contains implementations of monitoring decoding algorithms that use a base model and an expert model to eliminate the hallucinations in the LLM's generated output. The approach involves monitoring the generation process and switching between models based on confidence scores.

## Features

- **Multi-model decoding**: Support for base and expert model combinations
- **Multiple datasets**: GSM8K, TruthfulQA, TriviaQA, NQ-Open
- **Comprehensive evaluation**: EM metrics, accuracy scores, Truthfulness, Informativeness
- **Flexible configuration**: Command-line arguments for all parameters
- **Modular design**: Clean, reusable code structure

## Supported Models

- **Llama-2**: 7B, 13B, 70B variants
- **Llama-3**: 8B, 70B variants  
- **Gemma**: 2B, 27B variants

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd monitoring_decoding

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Text Generation

```bash
# Basic usage
python generate_text.py --data_task gsm8k --base_model Llama-2-7b --expert_model Llama-2-70b

# Custom parameters
python generate_text.py \
    --data_task truthfulqa \
    --base_model Llama-3-8b \
    --expert_model Llama-3-70b \
    --width 4 \
    --depth 4 \
    --branch 2 \
    --r 0.5 \
    --data_num 408
```

### Evaluation

#### GSM8K Evaluation
```bash
python evaluate_gsm8k.py \
    --base_model Llama-2-7b \
    --expert_model Llama-2-70b \
    --width 4 \
    --depth 3 \
    --branch 2 \
    --r 0.5 \
    --data_num 1319 \
    --save_results
```

#### EM Evaluation (for QA tasks)
```bash
python evaluation_em.py \
    --data_task nqopen \
    --base_model Llama-2-7b \
    --expert_model Llama-2-70b \
    --width 4 \
    --depth 3 \
    --branch 2 \
    --r 0.3 \
    --data_num 1000 \
    --save_results
```

## File Structure

```
monitoring_decoding/
├── generate_text.py          # Main generation script
├── evaluate_gsm8k.py         # GSM8K evaluation
├── evaluation_em.py          # EM evaluation for QA tasks
├── decoding.py           # Decoding algorithms
├── utils/
│   ├── generation_probs.py   # Probability calculations
│   ├── metric_utils.py       # Evaluation metrics
│   ├── tokenize.py           # Tokenization utilities
│   └── eval_info.txt         # Evaluation templates
├── results/                  # Generated outputs
├── evaluation_results/       # Evaluation results
└── README.md
```

## Parameters

### Generation Parameters
- `--data_task`: Dataset to use (gsm8k, truthfulqa, triviaqa, nqopen, xsum)
- `--base_model`: Base model name
- `--expert_model`: Expert model name
- `--width`: Search width for generation
- `--depth`: Length of generated candidates
- `--branch`: Number of candidates being remained
- `--r`: Ratio threshold for decoding
- `--data_num`: Number of data points to process

### Evaluation Parameters
- `--targets`: Models to evaluate (base, expert, decode)
- `--split_prediction`: Split prediction at newlines/periods
- `--save_results`: Save results to file
- `--output_dir`: Output directory for results

## Results

The evaluation scripts provide:
- **Accuracy scores** for each model
- **Statistical analysis** (mean, std, min, max)
- **Detailed breakdowns** of correct/incorrect predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{chang2025monitoringdecodingmitigatinghallucination,
      title={Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation}, 
      author={Yurui Chang and Bochuan Cao and Lu Lin},
      year={2025},
      eprint={2503.03106},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.03106}, 
}
```

