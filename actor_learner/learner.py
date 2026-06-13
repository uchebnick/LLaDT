import time
import math
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import cfg
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

def masked_logits(logits, mask, Z):
    B, L, V = logits.shape
    out = []
    for i in range(B):
        idx = mask[i].nonzero(as_tuple=True)[0]
        n = min(len(idx), Z)
        if n == 0:
            chunk = torch.zeros(Z, V, device=logits.device, dtype=logits.dtype)
        else:
            chunk = logits[i, idx[:n]]
            if n < Z:
                chunk = torch.cat([chunk, torch.zeros(Z-n, V, device=logits.device, dtype=logits.dtype)])
        out.append(chunk)
    return torch.stack(out)

def ce_on_mask(logits, ids, mask):
    B, L, V = logits.shape
    logits_shift = logits[:, :-1]
    targets = ids[:, 1:]
    ym = mask[:, 1:]
    valid_idx = ym.reshape(-1).nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    valid_logits = logits_shift.reshape(-1, V)[valid_idx]
    valid_targets = targets.reshape(-1)[valid_idx]
    mean_ce = F.cross_entropy(valid_logits.float(), valid_targets, reduction="mean")
    return mean_ce

def kl_balanced(q_logits, p_logits):
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    
    # Спортивный KL, где градиенты текут в обе стороны (симметричный)
    kl_qp = F.kl_div(p_log, q_log, reduction='none', log_target=True).sum(-1).mean()
    kl_pq = F.kl_div(q_log, p_log, reduction='none', log_target=True).sum(-1).mean()
    
    kl = 0.5 * (kl_qp + kl_pq)
    
    # Entropy of q: H(q) = -sum(q * log(q))
    q_prob = q_log.exp()
    entropy = -(q_prob * q_log).sum(dim=-1).mean()
    
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
        q_out = model(input_ids=t_ids, attention_mask=t_at)
        q_z_log = masked_logits(q_out.logits, t_zm_log, cfg.latent_len)
        del q_out
        
        # 2. p(z|Q) - Ученик (видит только Q)
        p_out = model(input_ids=s_ids, attention_mask=s_at)
        p_z_log = masked_logits(p_out.logits, s_zm_log, cfg.latent_len)
        del p_out
        
        # 3. KL Divergence (symmetric)
        kl, entropy = kl_balanced(q_z_log, p_z_log)

        # 4. Предсказание ответа из soft_z
        embed_matrix = model.get_input_embeddings().weight
        q_z_probs = F.gumbel_softmax(q_z_log.float(), tau=tau, hard=True, dim=-1).to(embed_matrix.dtype)
        soft_z_embeds = torch.matmul(q_z_probs, embed_matrix.detach())
        
        s_inputs_embeds = build_student_inputs_embeds(
            model, s_ids, s_zm_embed, s_qm, soft_z_embeds, cfg.learner_device
        )
        a_out = model(inputs_embeds=s_inputs_embeds, attention_mask=s_at)
        a_logits = a_out.logits
        del a_out
        
        ce = ce_on_mask(a_logits, s_ids, s_ym)
        
        total_loss = (ce + beta * kl) / cfg.grad_accum
        
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
                if sync_queue.full():
                    try: sync_queue.get_nowait()
                    except: pass
                sync_queue.put(sd)

        # Логирование
        if step % cfg.log_every == 0 and step > 0:
            mem_l = torch.cuda.max_memory_allocated(cfg.learner_device) / 1e9
            samples_processed = step * cfg.learner_batch_size
            logger.log_step(step, samples_processed, beta, lr, mem_l)
            
        if step % cfg.sample_every == 0:
            logger.log_sample(step, batch, z_list, a_logits.detach(), s_ym, tokenizer, beta)

    print("[Learner] Training finished.")
