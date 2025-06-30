import os
import json
import torch
import numpy as np
import argparse
import re
from typing import Dict, List, Any, Optional, Union

class GSM8KEvaluator:
    """Class for evaluating GSM8K mathematical reasoning tasks"""
    
    ANSWER_TRIGGER = "The answer is"
    INVALID_ANS = "[invalid]"
    
    @staticmethod
    def clean_answer(model_pred: str) -> Union[str, float]:
        """Clean and extract numerical answer from model prediction"""
        model_pred = model_pred.lower()
        preds = model_pred.split(GSM8KEvaluator.ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return GSM8KEvaluator.INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]

        return pred
    
    @staticmethod
    def evaluate_single_sample(sample: Dict[str, Any]) -> Dict[str, int]:
        """Evaluate a single sample for all models"""
        decode = GSM8KEvaluator.clean_answer(sample['decode'])
        if decode != GSM8KEvaluator.INVALID_ANS:
            decode = float(decode)

        base = GSM8KEvaluator.clean_answer(sample['base'])
        if base != GSM8KEvaluator.INVALID_ANS:
            base = float(base)
            
        expert = GSM8KEvaluator.clean_answer(sample['expert'])
        if expert != GSM8KEvaluator.INVALID_ANS:
            expert = float(expert)
            
        ans = float(sample['best'])
        
        return {
            'decode': int(decode == ans),
            'base': int(base == ans),
            'expert': int(expert == ans)
        }
    
    @staticmethod
    def evaluate_batch(generated_output: Dict[str, Any]) -> Dict[str, List[int]]:
        """Evaluate all samples in the generated output"""
        acc_decode = []
        acc_base = []
        acc_expert = []
        
        for k in generated_output.keys():
            sample = generated_output[k]
            results = GSM8KEvaluator.evaluate_single_sample(sample)
            
            acc_decode.append(results['decode'])
            acc_base.append(results['base'])
            acc_expert.append(results['expert'])
        
        return {
            'decode': acc_decode,
            'base': acc_base,
            'expert': acc_expert
        }
    
    @staticmethod
    def compute_metrics(accuracies: List[int]) -> Dict[str, float]:
        """Compute accuracy metrics from list of correct/incorrect predictions"""
        if not accuracies:
            return {'accuracy': 0.0, 'total_samples': 0}
        
        accuracy = np.mean(accuracies)
        total_samples = len(accuracies)
        
        return {
            'accuracy': accuracy,
            'total_samples': total_samples,
            'correct_predictions': sum(accuracies),
            'incorrect_predictions': total_samples - sum(accuracies)
        }

class ResultLoader:
    """Class for loading and managing generated output results"""
    
    @staticmethod
    def load_generated_output(data_task: str, base_model: str, expert_model: str,
                            width: int, depth: int, branch: int, r: float, 
                            data_num: int) -> Dict[str, Any]:
        """Load generated output from JSON file"""
        filename = "_".join([
            f'base={base_model}', f'exp={expert_model}', 
            f'width={width}', f'branch={branch}', 
            f'depth={depth}', f'r={r}', f'{data_num}', '.json'
        ])
        save_file_path = os.path.join("results", data_task, filename)
        
        with open(save_file_path, 'r') as file:
            return json.load(file)
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], filename: str) -> None:
        """Save evaluation metrics to JSON file"""
        with open(filename, 'w') as file:
            json.dump(metrics, file, indent=4)
        print(f"Metrics saved to {filename}")

class EvaluationRunner:
    """Main class for running GSM8K evaluations"""
    
    def __init__(self, data_task: str, base_model: str, expert_model: str,
                 width: int, depth: int, branch: int, r: float, data_num: int):
        self.data_task = data_task
        self.base_model = base_model
        self.expert_model = expert_model
        self.width = width
        self.depth = depth
        self.branch = branch
        self.r = r
        self.data_num = data_num
        
        # Load generated output
        self.generated_output = ResultLoader.load_generated_output(
            data_task, base_model, expert_model, width, depth, branch, r, data_num
        )
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run GSM8K evaluation"""
        print(f"Evaluating {len(self.generated_output)} samples...")
        
        # Evaluate all models
        accuracies = GSM8KEvaluator.evaluate_batch(self.generated_output)
        
        # Compute metrics for each model
        results = {}
        for model in ['base', 'expert', 'decode']:
            if model in accuracies:
                metrics = GSM8KEvaluator.compute_metrics(accuracies[model])
                results[model] = metrics
                print(f"{model} - Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results") -> str:
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"gsm8k_evaluation_{self.base_model}_{self.r}.json"
        filepath = os.path.join(output_dir, filename)
        
        ResultLoader.save_metrics(results, filepath)
        return filepath

def main():
    """Main function for running GSM8K evaluation"""
    parser = argparse.ArgumentParser(description="Run GSM8K evaluation on generated outputs")
    parser.add_argument('--data_task', type=str, default='gsm8k',
                       help='Dataset to evaluate (default: gsm8k)')
    parser.add_argument('--base_model', type=str, default='Llama-2-7b',
                       help='Base model name')
    parser.add_argument('--expert_model', type=str, default='Llama-2-70b',
                       help='Expert model name')
    parser.add_argument('--width', type=int, default=4, help='Search width')
    parser.add_argument('--depth', type=int, default=3, help='Depth')
    parser.add_argument('--branch', type=int, default=2, help='Branch factor')
    parser.add_argument('--r', type=float, default=0.5, help='Ratio threshold')
    parser.add_argument('--data_num', type=int, default=1319, help='Number of data points')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluation runner
    runner = EvaluationRunner(
        args.data_task, args.base_model, args.expert_model,
        args.width, args.depth, args.branch, args.r, args.data_num
    )
    
    # Run evaluation
    print(f"Running GSM8K evaluation...")
    print(f"Models: {args.base_model} -> {args.expert_model}")
    print(f"Parameters: width={args.width}, depth={args.depth}, r={args.r}")
    print("-" * 50)
    
    results = runner.run_evaluation()
    
    # Save results if requested
    if args.save_results:
        filepath = runner.save_results(results, args.output_dir)
        print(f"\nResults saved to: {filepath}")
    
    return results

if __name__ == "__main__":
    results = main()
    print("\nGSM8K evaluation completed!") 