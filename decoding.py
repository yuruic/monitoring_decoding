import torch
from utils.generation_probs import *
from utils.tokenize import tokenize_llama_chat, load_model_and_tokenizer, set_seed, tokenize_llama_instruct
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import functools
import operator
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import weakref
import threading
from contextlib import contextmanager
import inspect

class _ModelRegistry:
    """Singleton registry for model configurations"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
        return cls._instance
    
    def register(self, name: str, config: Dict[str, Any]) -> None:
        self._models[name] = config
    
    def get(self, name: str) -> Dict[str, Any]:
        return self._models.get(name, {})

class _DecodingStrategy(Enum):
    """Enumeration of decoding strategies"""
    GREEDY = "greedy"
    BEAM = "beam"
    SAMPLING = "sampling"
    MONITOR = "monitor"

@dataclass(frozen=True)
class _DecodingConfig:
    """Immutable configuration for decoding parameters"""
    depth: int = field(default=4)
    width: int = field(default=4)
    ratio_threshold: float = field(default=0.5)
    branch_factor: int = field(default=2)
    max_steps: int = field(default=40)
    
    def __post_init__(self):
        if self.ratio_threshold <= 0 or self.ratio_threshold > 1:
            raise ValueError("Ratio threshold must be in (0, 1]")

class _ModelLoaderMixin:
    """Mixin for model loading functionality"""
    
    @staticmethod
    def _create_model_loader() -> Callable:
        """Factory function for model loading strategies"""
        def _llama_loader(model_name: str, expert_name: str) -> Tuple[Any, Any, Any]:
            base_model, tokenizer = load_model_and_tokenizer(model_name=model_name)
            tokenizer.pad_token = tokenizer.eos_token
            expert_model, _ = load_model_and_tokenizer(model_name=expert_name)
            return base_model, expert_model, tokenizer
        
        def _gemma_loader(base_name: str, expert_name: str) -> Tuple[Any, Any, Any]:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name, device_map="auto", torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(base_name)
            tokenizer.pad_token = tokenizer.eos_token
            expert_model = AutoModelForCausalLM.from_pretrained(
                expert_name, device_map="auto", torch_dtype=torch.bfloat16
            )
            return base_model, expert_model, tokenizer
        
        return _llama_loader, _gemma_loader
    
    def _load_models(self, base_model_name: str) -> Tuple[Any, Any, Any]:
        """Load models based on configuration"""
        registry = _ModelRegistry()
        config = registry.get(base_model_name)
        
        if not config:
            # Default configurations
            configs = {
                'Llama-2-7b': {
                    'base': "meta-llama/Llama-2-7b-chat-hf",
                    'expert': "meta-llama/Llama-2-70b-chat-hf",
                    'loader': 'llama'
                },
                'Llama-3-8b': {
                    'base': "meta-llama/Llama-3.1-8B-Instruct",
                    'expert': "meta-llama/Llama-3.1-70B-Instruct",
                    'loader': 'llama'
                },
                'gemma-2b': {
                    'base': "google/gemma-2-2b-it",
                    'expert': "google/gemma-2-27b-it",
                    'loader': 'gemma'
                }
            }
            config = configs.get(base_model_name, {})
            registry.register(base_model_name, config)
        
        llama_loader, gemma_loader = self._create_model_loader()
        
        if config.get('loader') == 'llama':
            return llama_loader(config['base'], config['expert'])
        elif config.get('loader') == 'gemma':
            return gemma_loader(config['base'], config['expert'])
        else:
            raise ValueError(f"Unsupported model type: {base_model_name}")

def _performance_monitor(func: Callable) -> Callable:
    """Decorator for monitoring performance metrics"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        
        if hasattr(self, '_metrics'):
            self._metrics['execution_time'] = end_time - start_time
            self._metrics['call_count'] = self._metrics.get('call_count', 0) + 1
        
        return result
    return wrapper

def _cached_property(func: Callable) -> property:
    """Cached property decorator with weak references"""
    cache_name = f"_cached_{func.__name__}"
    
    def getter(self):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, func(self))
        return getattr(self, cache_name)
    
    def setter(self, value):
        setattr(self, cache_name, value)
    
    def deleter(self):
        if hasattr(self, cache_name):
            delattr(self, cache_name)
    
    return property(getter, setter, deleter)

class _DecodingEngine(ABC):
    """Abstract base class for decoding engines"""
    
    @abstractmethod
    def decode(self, input_tokens: torch.Tensor, config: _DecodingConfig) -> str:
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass

class _MonitorDecodingEngine(_DecodingEngine):
    """Concrete implementation of monitor decoding engine"""
    
    def __init__(self, base_model: Any, expert_model: Any, tokenizer: Any):
        self._base_model = base_model
        self._expert_model = expert_model
        self._tokenizer = tokenizer
        self._metrics = {}
        self._step_count = 0
        self._ratio_sum = 0.0
    
    @_performance_monitor
    def decode(self, input_tokens: torch.Tensor, config: _DecodingConfig) -> str:
        """Execute the monitor decoding algorithm"""
        fix_input_length = input_tokens.shape[1]
        current_tokens = input_tokens.clone()
        
        for step in range(config.max_steps):
            self._step_count += 1
            
            # Generate candidate tokens
            candidate_tokens = self._base_model.generate(
                current_tokens, 
                max_new_tokens=config.depth, 
                top_k=1, 
                do_sample=False
            )
            
            input_length = current_tokens.shape[1]
            
            # Calculate probability ratios
            with torch.no_grad():
                base_probs = generation_probs(self._base_model, candidate_tokens, input_length)
                expert_probs = generation_probs(self._expert_model, candidate_tokens, input_length)
                
                ratio = expert_probs / base_probs
                spec_prob = torch.exp(torch.sum(torch.log(ratio)))
                self._ratio_sum += spec_prob.item()
                
                # Decision logic
                if spec_prob.item() >= config.ratio_threshold:
                    current_tokens = candidate_tokens.clone()
                else:
                    current_tokens = self._execute_branching_strategy(
                        current_tokens, input_length, config
                    )
            
            # Early termination conditions
            if self._should_terminate(current_tokens, fix_input_length):
                break
        
        return self._finalize_output(current_tokens, fix_input_length)
    
    def _execute_branching_strategy(self, tokens: torch.Tensor, input_length: int, config: _DecodingConfig) -> torch.Tensor:
        """Execute the branching strategy for token selection"""
        for i in range(config.depth):
            k = config.width if i == 0 else config.width // config.branch_factor
            
            # Get top-k tokens
            logits = self._base_model(tokens).logits[:, -1, :]
            topk_probs, topk_tokens = torch.softmax(logits, dim=-1).topk(k)
            
            # Expand token sequences
            expanded_tokens = torch.cat((
                tokens.repeat(k, 1), 
                topk_tokens.T.flatten().unsqueeze(1)
            ), dim=1)
            
            # Calculate probabilities for expanded sequences
            base_probs = generation_probs(self._base_model, expanded_tokens, input_length)
            expert_probs = generation_probs(self._expert_model, expanded_tokens, input_length)
            
            # Select best candidates
            ratio = expert_probs / base_probs
            spec_prob = torch.exp(torch.sum(torch.log(ratio), dim=1))
            best_indices = spec_prob.topk(config.branch_factor).indices
            
            tokens = expanded_tokens[best_indices]
        
        # Return the best token sequence
        return expanded_tokens[spec_prob.argmax()].unsqueeze(0)
    
    def _should_terminate(self, tokens: torch.Tensor, fix_input_length: int) -> bool:
        """Check if decoding should terminate"""
        return (
            self._tokenizer.eos_token_id in tokens[:, fix_input_length:] or
            self._tokenizer.encode('\n\nQ:')[-2] in tokens[:, fix_input_length:-4-4+1]
        )
    
    def _finalize_output(self, tokens: torch.Tensor, fix_input_length: int) -> str:
        """Finalize the output by handling EOS tokens"""
        if self._tokenizer.eos_token_id in tokens:
            eos_positions = (tokens == self._tokenizer.eos_token_id).nonzero(as_tuple=True)[-1][-1]
            tokens = tokens[:, :(eos_positions+1)]
        
        return self._tokenizer.batch_decode(
            tokens[:, fix_input_length:], 
            skip_special_tokens=True
        )[0]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'execution_time': self._metrics.get('execution_time', 0),
            'step_count': self._step_count,
            'average_ratio': self._ratio_sum / max(self._step_count, 1),
            'call_count': self._metrics.get('call_count', 0)
        }

class _ModelFactory:
    """Factory class for creating model instances"""
    
    @staticmethod
    def create_decoder(base_model_name: str) -> '_MonitorDecodingEngine':
        """Create a decoder instance with the specified model"""
        loader = _ModelLoaderMixin()
        base_model, expert_model, tokenizer = loader._load_models(base_model_name)
        return _MonitorDecodingEngine(base_model, expert_model, tokenizer)

class monitor_decode:
    """Main interface class for monitor decoding"""
    
    def __init__(self, base_model_name: str):
        self._base_model_name = base_model_name
        self._engine = _ModelFactory.create_decoder(base_model_name)
        self._base_model = self._engine._base_model
        self._model_expert = self._engine._expert_model
        self.tokenizer = self._engine._tokenizer
        self._config = _DecodingConfig()
    
    @_cached_property
    def model_base(self):
        """Cached property for base model"""
        return self._base_model
    
    @_cached_property
    def model_expert(self):
        """Cached property for expert model"""
        return self._model_expert
    
    @property
    def time(self) -> Optional[float]:
        """Get execution time from metrics"""
        metrics = self._engine.get_metrics()
        return metrics.get('execution_time')
    
    @property
    def ratio(self) -> Optional[float]:
        """Get average ratio from metrics"""
        metrics = self._engine.get_metrics()
        return metrics.get('average_ratio')
    
    def decode_generate(self, input_tokens: torch.Tensor, depth: int = 4, 
                       width: int = 4, r: float = 0.5, branch: int = 2) -> str:
        """Generate decoded output using monitor decoding"""
        config = _DecodingConfig(
            depth=depth,
            width=width,
            ratio_threshold=r,
            branch_factor=branch
        )
        
        return self._engine.decode(input_tokens, config)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Cleanup if needed
        pass

# Legacy compatibility alias
dg_decode = monitor_decode
