import os
import math
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str        = "Qwen/Qwen3.5-0.8B"
    dataset_name: str      = "AI-MO/NuminaMath-CoT"
    max_train_samples: int = 50_000
    max_q_tokens: int      = 512
    max_a_tokens: int      = 48
    max_latent_len: int    = 128

    # ELBO / KL Balancing
    beta_start: float      = 0.001        # Начинаем с малого KL, чтобы Учитель научился извлекать ответ
    beta_end: float        = 1.0          # Плавно поднимаем штраф, чтобы заставить Ученика догонять
    beta_warmup: int       = 800          # Шагов для разогрева beta (ОЧЕНЬ медленно)
    
    target_kl: float       = 0.5          # Минимальный желаемый KL (чтобы вытолкнуть Учителя)
    gamma_kl: float        = 0.5          # Сила выталкивания (KL push)
    
    # Gumbel-Softmax
    tau_start: float       = 2.0
    tau_end: float         = 0.1
    tau_anneal_steps: int  = 400

    # Обучение
    actor_batch_size: int  = 16           # Огромный батч для генерации
    learner_batch_size: int = 8           # Увеличиваем батч в 2 раза (VRAM позволяет)
    grad_accum: int        = 4            # Эффективный батч = 32
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


import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import re
import queue

class MathQADataset(Dataset):
    def __init__(self, tokenizer, max_samples=None):
        raw = load_dataset(cfg.dataset_name, split="train")
        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))
        self.samples = []
        for item in raw:
            q = item.get("problem", "").strip()
            a = self._extract(item)
            if q and a:
                q_len = len(tokenizer(f"Q: {q}\n", add_special_tokens=False)["input_ids"])
                if q_len <= cfg.max_q_tokens:
                    self.samples.append({"q": q, "a": a})
        print(f"[Actor] Dataset loaded: {len(self.samples):,} examples")

    @staticmethod
    def _extract(item):
        if item.get("answer"): return str(item["answer"]).strip()
        boxes = re.findall(r"\\boxed\{([^}]+)\}", item.get("solution", ""))
        return boxes[-1].strip() if boxes else None

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def run_actor(rank, data_queue, sync_queue):
    print(f"[Actor] Starting on {cfg.actor_device}")
    
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype,
        device_map={"": cfg.actor_device}, trust_remote_code=True,
        attn_implementation="sdpa")
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(base_model, peft_config)
    model.eval()
    
    ds = MathQADataset(tokenizer, max_samples=cfg.max_train_samples)
    loader = DataLoader(ds, batch_size=cfg.actor_batch_size, shuffle=True, drop_last=True)
    
    pad = tokenizer.pad_token_id
    
    step = 0
    while True:
        for batch in loader:
            # 1. Проверяем новые веса
            try:
                state_dict = sync_queue.get_nowait()
                model.load_state_dict(state_dict, strict=False)
                if step % 10 == 0:
                    print(f"[Actor] Received updated LoRA weights at step {step}")
            except queue.Empty:
                pass

            t0 = time.time()
            prompts = []
            B = len(batch["q"]) if isinstance(batch, dict) else len(batch)
            def tok(t): return tokenizer(t, add_special_tokens=False)["input_ids"]
            TS = tok("<think>\n"); AP = tok("\nA: ")
            bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
            
            for i in range(B):
                q = batch["q"][i] if isinstance(batch, dict) else batch[i]["q"]
                a = batch["a"][i] if isinstance(batch, dict) else batch[i]["a"]
                
                q_ids = tok(f"Q: {q}\n")[:cfg.max_q_tokens]
                a_ids = tok(a)[:cfg.max_a_tokens]
                
                ids = bos + q_ids + AP + a_ids + TS
                prompts.append(ids)
                
            mp_len = max(len(p) for p in prompts)
            gids = torch.full((B, mp_len), pad, dtype=torch.long, device=cfg.actor_device)
            gmsk = torch.zeros(B, mp_len, dtype=torch.long, device=cfg.actor_device)
            for i, p in enumerate(prompts):
                gids[i, mp_len-len(p):] = torch.tensor(p, dtype=torch.long)
                gmsk[i, mp_len-len(p):] = 1
                
            with torch.no_grad():
                gen = model.generate(
                    input_ids=gids, attention_mask=gmsk,
                    max_new_tokens=cfg.max_latent_len,
                    do_sample=True, temperature=1.0,
                    pad_token_id=pad, eos_token_id=tokenizer.eos_token_id, 
                    use_cache=True)
                    
            z_list = []
            for i in range(B):
                z = gen[i, mp_len:].cpu().tolist()
                if tokenizer.eos_token_id in z:
                    eos_idx = z.index(tokenizer.eos_token_id)
                    z = z[:eos_idx+1]
                if len(z) < cfg.max_latent_len:
                    z += [pad] * (cfg.max_latent_len - len(z))
                z_list.append(z[:cfg.max_latent_len])
                
            # Кладем сэмплы в очередь по одному или батчами
            # Чтобы Learner мог брать их маленькими порциями (learner_batch_size)
            # мы разобьем сгенерированный батч на отдельные сэмплы
            for i in range(B):
                item = {"q": batch["q"][i] if isinstance(batch, dict) else batch[i]["q"], 
                        "a": batch["a"][i] if isinstance(batch, dict) else batch[i]["a"]}
                while True:
                    try:
                        data_queue.put((item, z_list[i]), timeout=60.0)
                        break
                    except queue.Full:
                        pass # Ждем пока Learner заберет данные
                
            t1 = time.time()
            if step % 5 == 0:
                print(f"[Actor] Generated {B} samples in {t1-t0:.2f}s | Queue size: ~{data_queue.qsize()}", flush=True)
            step += 1


import time
import math
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

import queue

# =============================================================================
# Утилиты логирования и метрик
# =============================================================================

def get_beta(step: int) -> float:
    if step >= cfg.beta_warmup:
        return cfg.beta_end
    progress = step / max(1, cfg.beta_warmup)
    return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * progress

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

def fmt_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

def fmt_bar(step, total, width=20):
    filled = int(width * step / max(total, 1))
    return "█" * filled + "░" * (width - filled)

class Logger:
    def __init__(self):
        self.reset()
        self._t0 = time.time()
        self._step_times = []

    def reset(self):
        self.s = {"loss": 0, "ce": 0, "kl": 0, "ent": 0}
        self._n = 0

    def update(self, loss, ce, kl, ent, step_t):
        self.s["loss"] += loss
        self.s["ce"] += ce
        self.s["kl"] += kl
        self.s["ent"]  += ent
        self._n         += 1
        self._step_times.append(step_t)

    def log_step(self, step, samples, beta, lr, mem_l):
        n = self._n or 1
        avg_t = sum(self._step_times[-20:]) / len(self._step_times[-20:]) if self._step_times else 0
        eta   = avg_t * (cfg.max_steps - step)
        elapsed = time.time() - self._t0

        bar = fmt_bar(step, cfg.max_steps)
        pct = 100 * step / cfg.max_steps

        print(
            f"\r[{bar}] {pct:4.1f}%  samples={samples}  step={step}/{cfg.max_steps}"
            f"  loss={self.s['loss']/n:.3f}"
            f"  ce={self.s['ce']/n:.3f}  kl={self.s['kl']/n:.3f}"
            f"  ent={self.s['ent']/n:.3f}"
            f"  β={beta:.2f}  lr={lr:.1e}"
            f"  {avg_t:.2f}s/it  ⏱{fmt_time(elapsed)}  ETA={fmt_time(eta)}"
            f"  VRAM=[{mem_l:.1f}]G",
            flush=True
        )
        self.reset()

    def log_sample(self, step, batch, z_list, s_logits, s_ym, tokenizer, beta):
        z0     = z_list[0]
        z_text = tokenizer.decode(z0, skip_special_tokens=False)
        uniq   = len(set(z0))
        top    = Counter(z0).most_common(8)

        y_idx = s_ym[0].nonzero(as_tuple=True)[0]
        pred_a = ""
        if len(y_idx) > 0:
            pred_ids = s_logits[0, y_idx[0]-1:y_idx[-1]].argmax(-1)
            pred_a   = tokenizer.decode(pred_ids.cpu(), skip_special_tokens=True)

        readable = sum(1 for t in z0
                       if tokenizer.decode([t]).strip()
                       and tokenizer.decode([t]).strip()[0].isalpha())
        readability = readable / max(len(z0), 1)

        print(f"\n{'━'*68}")
        print(f"  📊 Z-SAMPLE   step={step}   β={beta:.2f}")
        print(f"{'━'*68}")
        print(f"  Q  : {batch[0]['q'][:110]}")
        print(f"  A✓ : {batch[0]['a'][:70]}")
        print(f"  Â  : {pred_a[:70]}")
        print(f"{'─'*68}")
        print(f"  z  : {z_text[:260]}")
        print(f"{'─'*68}")
        print(f"  unique={uniq}/{cfg.latent_len}   "
              f"readability={readability:.1%}   "
              f"(↓ хорошо — токены становятся латентными)")
        top_str = "  ".join(f"'{tokenizer.decode([t])}'×{c}" for t, c in top)
        print(f"  top: {top_str}")
        print(f"{'━'*68}\n")

# =============================================================================
# Математика и последовательности
# =============================================================================

def _mask(length, start, end):
    m = [False] * length
    for j in range(start, min(end, length)): m[j] = True
    return m

def build_sequences(batch, tokenizer, z_list, device):
    def tok(t): return tokenizer(t, add_special_tokens=False)["input_ids"]

    TS = tok("<think>\n"); TE = tok("\n</think>\n"); AP = tok("\nA: ")
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []

    t_seqs, t_zm_log, t_ym = [], [], []
    s_seqs, s_zm_log, s_zm_embed, s_ym, s_qm = [], [], [], [], []

    for i, item in enumerate(batch):
        q_ids = tok(f"Q: {item['q']}\n")[:cfg.max_q_tokens]
        a_ids = tok(item["a"])[:cfg.max_a_tokens]
        z     = z_list[i]

        t = bos + q_ids + AP + a_ids + TS + z
        tz_s = len(bos) + len(q_ids) + len(AP) + len(a_ids) + len(TS)
        tz_e = tz_s + len(z)
        ty_s = len(bos) + len(q_ids) + len(AP)
        ty_e = ty_s + len(a_ids)
        t_seqs.append(t)
        t_zm_log.append(_mask(len(t), tz_s - 1, tz_e - 1))
        t_ym.append(_mask(len(t), ty_s, ty_e))

        s = bos + q_ids + TS + z + TE + AP + a_ids
        sz_s = len(bos) + len(q_ids) + len(TS)
        sz_e = sz_s + len(z)
        sy_s = sz_e + len(TE) + len(AP)
        sy_e = sy_s + len(a_ids)
        s_seqs.append(s)
        s_zm_log.append(_mask(len(s), sz_s - 1, sz_e - 1))
        s_zm_embed.append(_mask(len(s), sz_s, sz_e))
        s_ym.append(_mask(len(s), sy_s, sy_e))
        s_qm.append(_mask(len(s), len(bos), len(bos) + len(q_ids)))

    def _pad(seqs, masks, dev):
        L = max(len(s) for s in seqs); B = len(seqs)
        pad = tokenizer.pad_token_id
        ids = torch.full((B, L), pad, dtype=torch.long)
        attn = torch.zeros(B, L, dtype=torch.long)
        ms = [torch.zeros(B, L, dtype=torch.bool) for _ in masks]
        for i, (s, *mm) in enumerate(zip(seqs, *masks)):
            n = len(s)
            ids[i, :n] = torch.tensor(s, dtype=torch.long)
            attn[i, :n] = 1
            for k, m in enumerate(mm):
                ms[k][i, :n] = torch.tensor(m, dtype=torch.bool)
        return ids.to(dev), attn.to(dev), [m.to(dev) for m in ms]

    t_ids, t_at, (tzm_log, tym) = _pad(t_seqs, [t_zm_log, t_ym], device)
    s_ids, s_at, (szm_log, szm_embed, sym, sqm) = _pad(s_seqs, [s_zm_log, s_zm_embed, s_ym, s_qm], device)
    return (t_ids, t_at, tzm_log, tym), (s_ids, s_at, szm_log, szm_embed, sym, sqm)

def build_student_inputs_embeds(model, s_ids, s_zm_embed, s_qm, soft_z, device):
    embed_matrix = model.get_input_embeddings().weight
    s_ids = s_ids.to(device)
    s_embeds = embed_matrix[s_ids].clone() # [B, L, D]
    
    # Заменяем токены z на soft_z
    for i in range(len(s_ids)):
        z_idx = s_zm_embed[i].nonzero(as_tuple=True)[0]
        s_embeds[i, z_idx] = soft_z[i, :len(z_idx)]
        
    return s_embeds

def masked_hidden(hidden, mask, max_len):
    B, L, D = hidden.shape
    out = []
    for i in range(B):
        idx = mask[i].nonzero(as_tuple=True)[0]
        n = min(len(idx), max_len)
        if n == 0:
            chunk = torch.zeros(max_len, D, device=hidden.device, dtype=hidden.dtype)
        else:
            chunk = hidden[i, idx[:n]]
            if n < max_len:
                chunk = torch.cat([chunk, torch.zeros(max_len-n, D, device=hidden.device, dtype=hidden.dtype)])
        out.append(chunk)
    return torch.stack(out)

def ce_on_mask(logits, ids, mask):
    B, L, V = logits.shape
    valid_targets = []
    valid_logits = []
    for i in range(B):
        idx = mask[i].nonzero(as_tuple=True)[0]
        # Shifted targets
        target_idx = idx + 1
        valid_mask = target_idx < ids.shape[1]
        target_idx = target_idx[valid_mask]
        valid_targets.append(ids[i, target_idx])
        
        n = len(target_idx)
        if n > 0:
            valid_logits.append(logits[i, :n])
    
    valid_targets = torch.cat(valid_targets)
    if len(valid_targets) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    valid_logits = torch.cat(valid_logits, dim=0)
    return F.cross_entropy(valid_logits.float(), valid_targets, reduction="mean")

def kl_balanced_masked(q_logits, p_logits, mask):
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    
    # Calculate KL token-wise, keep sequence dimension: [B, L_z]
    kl_qp = F.kl_div(p_log, q_log, reduction='none', log_target=True).sum(-1)
    kl_pq = F.kl_div(q_log, p_log, reduction='none', log_target=True).sum(-1)
    kl = 0.5 * (kl_qp + kl_pq)
    
    # Scale by detached mask to avoid length collapse
    mask_detached = mask.detach()
    kl = (kl * mask_detached).sum() / mask_detached.sum().clamp(min=1.0)
    
    # Entropy
    q_prob = q_log.exp()
    entropy = -(q_prob * q_log).sum(dim=-1)
    entropy = (entropy * mask_detached).sum() / mask_detached.sum().clamp(min=1.0)
    
    return kl, entropy

# =============================================================================
# Основной процесс
# =============================================================================

def run_learner(rank, data_queue, sync_queue):
    print(f"[Learner] Starting on {cfg.learner_device}")
    
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype,
        device_map={"": cfg.learner_device}, trust_remote_code=True,
        attn_implementation="sdpa")
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.train(); model.config.use_cache = False
    
    import bitsandbytes as bnb
    opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    logger = Logger()
    
    for step in range(cfg.max_steps):
        t0 = time.time()
        beta = get_beta(step)
        tau = get_tau(step)
        
        batch = []
        z_list = []
        while len(batch) < cfg.learner_batch_size:
            try:
                item, z = data_queue.get(timeout=300.0)
                batch.append(item)
                z_list.append(z)
            except queue.Empty:
                print("[Learner] Queue empty for 300s. Actor might have crashed. Exiting.")
                import sys; sys.exit(1)
            
        (t_ids, t_at, t_zm_log, t_ym),         (s_ids, s_at, s_zm_log, s_zm_embed, s_ym, s_qm) = build_sequences(batch, tokenizer, z_list, cfg.learner_device)
        
        # 1. q(z|Q,A) - Учитель (видит Q+A)
        q_out = model(input_ids=t_ids, attention_mask=t_at, output_hidden_states=True)
        q_z_hid = masked_hidden(q_out.hidden_states[-1], t_zm_log, cfg.max_latent_len)
        q_z_log = model.get_output_embeddings()(q_z_hid)
        del q_out
        
        # Calculate Cumulative Probability Mask based on EOS
        eos_id = tokenizer.eos_token_id
        q_probs = F.softmax(q_z_log.float(), dim=-1)
        p_eos = q_probs[:, :, eos_id]
        p_keep = 1.0 - p_eos
        cum_keep = torch.cumprod(p_keep, dim=1)
        
        z_mask = torch.ones_like(p_keep)
        z_mask[:, 1:] = cum_keep[:, :-1]
        
        # 2. p(z|Q) - Ученик (видит только Q)
        p_out = model(input_ids=s_ids, attention_mask=s_at, output_hidden_states=True)
        p_z_hid = masked_hidden(p_out.hidden_states[-1], s_zm_log, cfg.max_latent_len)
        p_z_log = model.get_output_embeddings()(p_z_hid)
        del p_out
        
        # 3. KL Divergence (symmetric)
        kl, entropy = kl_balanced_masked(q_z_log, p_z_log, z_mask)

        # Выталкивающий штраф для Учителя: заставляем его отличаться от Ученика минимум на target_kl
        kl_teacher_only, _ = kl_balanced_masked(q_z_log, p_z_log.detach(), z_mask)
        kl_push_loss = cfg.gamma_kl * torch.relu(cfg.target_kl - kl_teacher_only)

        # 4. Предсказание ответа из soft_z
        embed_matrix = model.get_input_embeddings().weight
        q_z_probs = F.gumbel_softmax(q_z_log.float(), tau=tau, hard=True, dim=-1).to(embed_matrix.dtype)
        soft_z_embeds = torch.matmul(q_z_probs, embed_matrix.detach())
        
        # Scale embeddings by mask to stop gradients and attention for padded tokens
        soft_z_embeds = soft_z_embeds * z_mask.unsqueeze(-1)
        
        s_inputs_embeds = build_student_inputs_embeds(
            model, s_ids, s_zm_embed, s_qm, soft_z_embeds, cfg.learner_device
        )
        a_out = model(inputs_embeds=s_inputs_embeds, attention_mask=s_at, output_hidden_states=True)
        a_hid = masked_hidden(a_out.hidden_states[-1], s_ym, cfg.max_a_tokens)
        a_logits = model.get_output_embeddings()(a_hid)
        del a_out
        
        ce = ce_on_mask(a_logits, s_ids, s_ym)
        
        total_loss = (ce + beta * kl + kl_push_loss) / cfg.grad_accum
        
        ce_val = ce.item()
        kl_val = kl.item()
        ent_val = entropy.item()
        loss_val = total_loss.item() * cfg.grad_accum
        
        # Освобождаем память до backward
        del ce, kl, entropy, q_z_log, p_z_log, q_z_probs, soft_z_embeds, s_inputs_embeds
        
        total_loss.backward()
        
        logger.update(
            loss_val,
            ce_val, kl_val,
            ent_val,
            time.time() - t0
        )
        
        if (step + 1) % cfg.grad_accum == 0:
            lr = get_lr(step)
            for g in opt.param_groups: g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step(); opt.zero_grad()
            
            # Синхронизация весов (передаем одну модель)
            if (step // cfg.grad_accum) % cfg.sync_every_n_steps == 0:
                sd = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
                while True:
                    try: sync_queue.get_nowait()
                    except queue.Empty: break
                try:
                    sync_queue.put_nowait(sd)
                except queue.Full:
                    pass

        # Логирование
        if step % cfg.log_every == 0 and step > 0:
            mem_l = torch.cuda.max_memory_allocated(cfg.learner_device) / 1e9
            samples_processed = step * cfg.learner_batch_size
            logger.log_step(step, samples_processed, beta, lr, mem_l)
            
        if step % cfg.sample_every == 0:
            logger.log_sample(step, batch, z_list, a_logits.detach(), s_ym, tokenizer, beta)

    print("[Learner] Training finished.")


import os
import sys
import torch
import torch.multiprocessing as mp



def main():
    # Настройка метода старта процесса (важно для CUDA)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    print("🚀 Запуск асинхронного Actor-Learner пайплайна...")
    print(f"Actor GPU: {cfg.actor_device} | Learner GPU: {cfg.learner_device}")
    
    # Очереди с ограничением размера (очередь сэмплов)
    data_queue = mp.Queue(maxsize=cfg.data_queue_max_size)
    
    # Очередь для синхронизации весов от Learner к Actor
    sync_queue = mp.Queue(maxsize=2)
    
    # Импортируем внутри, чтобы не инициализировать CUDA в главном процессе
    
    
    
    actor_p = mp.Process(target=run_actor, args=(0, data_queue, sync_queue))
    learner_p = mp.Process(target=run_learner, args=(1, data_queue, sync_queue))
    
    actor_p.start()
    learner_p.start()
    
    try:
        learner_p.join()  # Ждем, пока Learner закончит обучение (max_steps)
    except KeyboardInterrupt:
        print("Остановка процессов...")
    finally:
        actor_p.terminate()
        learner_p.terminate()
        actor_p.join()
        
    print("✅ Обучение завершено.")


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
        os.system("pip install bitsandbytes accelerate datasets git+https://github.com/huggingface/transformers.git wandb peft")
    main()
