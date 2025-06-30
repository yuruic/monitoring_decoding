import os
import json
import numpy as np
import argparse
from typing import Dict, List, Any, Optional
import regex as re
from utils.metric_utils import best_em, best_subspan_em

class EMMetrics:
    """Class for computing Exact Match (EM) metrics"""
    
    @staticmethod
    def compute_metrics(prediction: str, refs: List[str]) -> Dict[str, float]:
        """Compute EM and Subspan EM metrics for a prediction against references"""
        # Placeholder - no calculations
        return {"EM": 0.0, "Subspan_EM": 0.0}
    
    @staticmethod
    def evaluate_target(generated_output: Dict[str, Any], target: str, 
                       split_prediction: bool = False) -> Dict[str, float]:
        """Evaluate a specific target (base, expert, decode) using EM metrics"""
        # Placeholder - no calculations
        return {
            "EM": 0.0,
            "Subspan_EM": 0.0,
        }

class ResultLoader:
    """Class for loading and managing generated output results"""
    
    @staticmethod
    def load_generated_output(data_task: str, base_model: str, expert_model: str,
                            width: int, depth: int, 
                            branch: int, r: float, data_num: int) -> Dict[str, Any]:
        """Load generated output from JSON file"""
        filename = "_".join([
            f'base={base_model}', f'exp={expert_model}', 
            f'width={width}', f'branch={branch}', 
            f'depth={depth}', f'r={r}',
            f'{data_num}', '.json'
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
    """Main class for running EM evaluations"""
    
    def __init__(self, data_task: str, base_model: str, expert_model: str,
                 width: int, depth: int,
                 branch: int, r: float, data_num: int):
        self.data_task = data_task
        self.base_model = base_model
        self.expert_model = expert_model
        self.width = width
        self.depth2 = depth
        self.branch = branch
        self.r = r
        self.data_num = data_num
        
        # Load generated output
        self.generated_output = ResultLoader.load_generated_output(
            data_task, base_model, expert_model, width, depth, branch, r, data_num
        )
    
    def run_evaluation(self, targets: List[str] = None, split_prediction: bool = False) -> Dict[str, Any]:
        """Run EM evaluation for specified targets"""
        if targets is None:
            targets = ['base', 'expert', 'decode']
        
        results = {}
        
        # Evaluate each target
        for target in targets:
            if target in self.generated_output[list(self.generated_output.keys())[0]]:
                print(f"Evaluating {target}...")
                metrics = EMMetrics.evaluate_target(self.generated_output, target, split_prediction)
                results[target] = metrics
                print(f"{target} - EM: {metrics['EM']:.4f}, Subspan_EM: {metrics['Subspan_EM']:.4f}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results") -> None:
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"em_evaluation_{self.data_task}_{self.base_model}_{self.r}.json"
        filepath = os.path.join(output_dir, filename)
        
        ResultLoader.save_metrics(results, filepath)
        return filepath

def main():
    """Main function for running EM evaluation"""
    parser = argparse.ArgumentParser(description="Run EM evaluation on generated outputs")
    parser.add_argument('--data_task', type=str, default='nqopen',
                       choices=['triviaqa', 'nqopen', 'truthfulqa', 'gsm8k'],
                       help='Dataset to evaluate')
    parser.add_argument('--base_model', type=str, default='Llama-2-7b',
                       help='Base model name')
    parser.add_argument('--expert_model', type=str, default='Llama-2-70b',
                       help='Expert model name')
    parser.add_argument('--width', type=int, default=4, help='Search width')
    parser.add_argument('--depth', type=int, default=3, help='Depth 1')
    parser.add_argument('--branch', type=int, default=2, help='Branch factor')
    parser.add_argument('--r', type=float, default=0.3, help='Ratio threshold')
    parser.add_argument('--data_num', type=int, default=1000, help='Number of data points')
    parser.add_argument('--targets', nargs='+', default=['base', 'expert', 'decode'],
                       help='Targets to evaluate')
    parser.add_argument('--split_prediction', action='store_true',
                       help='Split prediction at newlines, periods, or commas')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluation runner
    runner = EvaluationRunner(
        args.data_task, args.base_model, args.expert_model,
        args.width, args.depth, 
        args.branch, args.r, args.data_num
    )
    
    # Run evaluation
    print(f"Running EM evaluation for {args.data_task} dataset...")
    print(f"Models: {args.base_model} -> {args.expert_model}")
    print(f"Parameters: width={args.width}, max_width={args.max_width}, r={args.r}")
    print("-" * 50)
    
    results = runner.run_evaluation(args.targets, args.split_prediction)
    
    # Save results if requested
    if args.save_results:
        filepath = runner.save_results(results, args.output_dir)
        print(f"\nResults saved to: {filepath}")
    
    return results

if __name__ == "__main__":
    results = main()
    print("\nEvaluation completed!") 