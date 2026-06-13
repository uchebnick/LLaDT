import os
import math
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str        = "Qwen/Qwen3.5-0.8B"
    dataset_name: str      = "AI-MO/NuminaMath-CoT"
    max_train_samples: int = 50_000
    max_q_tokens: int      = 192
    max_a_tokens: int      = 48
    latent_len: int        = 64

    # ELBO / KL Balancing
    beta: float            = 5.0          # KL divergence penalty
    
    # Gumbel-Softmax
    tau_start: float       = 2.0
    tau_end: float         = 0.1
    tau_anneal_steps: int  = 400

    # Обучение
    actor_batch_size: int  = 16           # Огромный батч для генерации
    learner_batch_size: int = 4           # Уменьшаем, чтобы избежать OOM
    grad_accum: int        = 8            # Увеличиваем, чтобы эффективный батч остался 32
    lr: float              = 3e-4
    max_steps: int         = 1000
    warmup_steps: int      = 100
    max_grad_norm: float   = 1.0

    # LoRA
    lora_r: int            = 16
    lora_alpha: int        = 32
    lora_dropout: float    = 0.05

    # Железо (указываются устройства)
    dtype: str             = "bfloat16"
    actor_device: str      = "cuda:0"
    learner_device: str    = "cuda:1"

    # Queue Settings
    data_queue_max_size: int = 100        # Replay Buffer
    sync_every_n_steps: int  = 5          # Синхронизация весов LoRA от Learner к Actor

    # Логирование
    log_every: int         = 10
    sample_every: int      = 30
    save_every: int        = 500
    output_dir: str        = "./checkpoints_async"

cfg = Config()

def get_beta(step: int) -> float:
    return cfg.beta

def get_tau(step: int) -> float:
    if step >= cfg.tau_anneal_steps:
        return cfg.tau_end
    progress = step / max(1, cfg.tau_anneal_steps)
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * progress

def get_lr(step: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * p))
