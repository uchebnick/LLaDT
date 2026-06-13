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
    beta: float            = 5.0          # Огромный штраф, чтобы Учитель боялся шифров
    kl_balance_alpha: float = 0.15        # Доля градиента для Студента (Учитель получает 1.0)
    
    # Gumbel-Softmax
    tau_start: float       = 2.0
    tau_end: float         = 0.5
    tau_anneal_steps: int  = 400

    # Обучение
    actor_batch_size: int  = 16           # Огромный батч для генерации (ведь там только Учитель!)
    learner_batch_size: int = 8           # Увеличиваем батч для Forward/Backward (было 2)
    grad_accum: int        = 4            # Уменьшаем, чтобы сохранить эффективный батч 32 (8*4=32)
    lr: float              = 1e-4
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
    
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype,
        device_map={"": cfg.actor_device}, trust_remote_code=True,
        attn_implementation="sdpa")
    teacher = get_peft_model(teacher, lora_cfg)
    teacher.eval()
    
    ds = MathQADataset(tokenizer, max_samples=cfg.max_train_samples)
    # Используем бесконечный загрузчик, потому что Actor крутится пока жив Learner
    loader = DataLoader(ds, batch_size=cfg.actor_batch_size, shuffle=True, drop_last=True)
    
    gen_temp = 0.6
    pad = tokenizer.pad_token_id
    
    step = 0
    while True:
        for batch in loader:
            # 1. Проверяем синхронизацию весов
            if not sync_queue.empty():
                try:
                    new_state_dict = sync_queue.get_nowait()
                    teacher.load_state_dict(new_state_dict, strict=False)
                    print(f"[Actor] Received updated LoRA weights at step {step}")
                except Exception:
                    pass

            t0 = time.time()
            prompts = []
            B = len(batch["q"]) if isinstance(batch, dict) else len(batch)
            for i in range(B):
                q = batch["q"][i] if isinstance(batch, dict) else batch[i]["q"]
                a = batch["a"][i] if isinstance(batch, dict) else batch[i]["a"]
                text = f"Q: {q[:400]}\nA: {a[:80]}\n<think>\n"
                ids = tokenizer(text, add_special_tokens=True)["input_ids"]
                prompts.append(ids[:cfg.max_q_tokens + cfg.max_a_tokens + 10])
                
            mp_len = max(len(p) for p in prompts)
            gids = torch.full((B, mp_len), pad, dtype=torch.long, device=cfg.actor_device)
            gmsk = torch.zeros(B, mp_len, dtype=torch.long, device=cfg.actor_device)
            for i, p in enumerate(prompts):
                gids[i, mp_len-len(p):] = torch.tensor(p, dtype=torch.long)
                gmsk[i, mp_len-len(p):] = 1
                
            with torch.no_grad():
                gen = teacher.generate(
                    input_ids=gids, attention_mask=gmsk,
                    max_new_tokens=cfg.latent_len, min_new_tokens=cfg.latent_len,
                    do_sample=True, temperature=gen_temp, top_p=0.9,
                    pad_token_id=pad, use_cache=True)
                    
            z_list = []
            for i in range(B):
                z = gen[i, mp_len:mp_len+cfg.latent_len].cpu().tolist()
                if len(z) < cfg.latent_len:
                    z += [pad] * (cfg.latent_len - len(z))
                z_list.append(z[:cfg.latent_len])
                
            # Кладем сэмплы в очередь по одному или батчами
            # Чтобы Learner мог брать их маленькими порциями (learner_batch_size)
            # мы разобьем сгенерированный батч на отдельные сэмплы
            import queue
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


# =============================================================================
# Утилиты логирования и метрик
# =============================================================================

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
        self.s = dict(loss=0., s_ce=0., s_kl=0., t_ce=0., t_kl=0.)
        self._n = 0

    def update(self, loss, s_ce, s_kl, t_ce, t_kl, step_t):
        self.s["loss"]  += loss
        self.s["s_ce"]  += s_ce
        self.s["s_kl"]  += s_kl
        self.s["t_ce"]  += t_ce
        self.s["t_kl"]  += t_kl
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
            f"  s_ce={self.s['s_ce']/n:.3f}  s_kl={self.s['s_kl']/n:.3f}"
            f"  t_ce={self.s['t_ce']/n:.3f}  t_kl={self.s['t_kl']/n:.3f}"
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

        t = bos + q_ids + a_ids + TS + z
        tz_s = len(bos) + len(q_ids) + len(a_ids) + len(TS)
        tz_e = tz_s + len(z)
        ty_s = len(bos) + len(q_ids)
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

def build_student_inputs_embeds(student, s_input_ids, s_z_mask, soft_z_embeds, device):
    embed_layer = student.get_input_embeddings()
    base_embeds = embed_layer(s_input_ids)
    result = base_embeds.clone()
    result[s_z_mask] = soft_z_embeds.reshape(-1, base_embeds.size(-1))
    return result

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
    kl_student = F.kl_div(p_log, q_log.detach(), reduction='none', log_target=True).sum(-1).mean()
    kl_teacher = F.kl_div(q_log, p_log.detach(), reduction='none', log_target=True).sum(-1).mean()
    return kl_student, kl_teacher

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
    
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype,
        device_map={"": cfg.learner_device}, trust_remote_code=True,
        attn_implementation="sdpa")
    teacher = get_peft_model(teacher, lora_cfg)
    teacher.gradient_checkpointing_enable()
    teacher.train(); teacher.config.use_cache = False
    
    student = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype,
        device_map={"": cfg.learner_device}, trust_remote_code=True,
        attn_implementation="sdpa")
    student = get_peft_model(student, lora_cfg)
    student.gradient_checkpointing_enable()
    student.train(); student.config.use_cache = False

    # Используем AdamW от PyTorch напрямую (без bitsandbytes для совместимости mp) 
    # или можно оставить bnb
    import bitsandbytes as bnb
    opt_t = bnb.optim.PagedAdamW8bit(teacher.parameters(), lr=cfg.lr, weight_decay=0.01)
    opt_s = bnb.optim.PagedAdamW8bit(student.parameters(), lr=cfg.lr, weight_decay=0.01)

    logger = Logger()
    
    for step in range(cfg.max_steps):
        t0 = time.time()
        beta = get_beta(step)
        tau = get_tau(step)
        
        # 1. Извлекаем батч из очереди (по размеру learner_batch_size)
        batch = []
        z_list = []
        import queue
        # Блокируемся, пока не наберем батч, с таймаутом чтобы избежать дедлока
        while len(batch) < cfg.learner_batch_size:
            try:
                item, z = data_queue.get(timeout=60.0)
                batch.append(item)
                z_list.append(z)
            except queue.Empty:
                print("[Learner] Queue empty for 60s. Actor might have crashed. Exiting.")
                import sys; sys.exit(1)
            
        # 2. Строим последовательности
        (t_ids, t_at, t_zm_log, t_ym), \
        (s_ids, s_at, s_zm_log, s_zm_embed, s_ym, s_qm) = build_sequences(batch, tokenizer, z_list, cfg.learner_device)
        
        # 3. Градиентные накопления (цикл внутри батча, если мы хотим имитировать grad_accum)
        # В нашем случае мы просто пропускаем loss через backward() и делаем step() реже
        
        t_out = teacher(input_ids=t_ids, attention_mask=t_at)
        t_logits = t_out.logits
        t_z_log = masked_logits(t_logits, t_zm_log, cfg.latent_len)
        
        s_embed_matrix = student.get_input_embeddings().weight
        t_z_probs = F.gumbel_softmax(t_z_log.float(), tau=tau, hard=True, dim=-1).to(s_embed_matrix.dtype)
        # Поскольку обе модели на learner_device, обойдемся без CrossDeviceCopy
        soft_z_embeds = torch.matmul(t_z_probs, s_embed_matrix.detach())
        
        s_inputs_embeds = build_student_inputs_embeds(
            student, s_ids, s_zm_embed, soft_z_embeds, cfg.learner_device
        )
        s_out = student(inputs_embeds=s_inputs_embeds, attention_mask=s_at)
        s_logits = s_out.logits
        s_z_log = masked_logits(s_logits, s_zm_log, cfg.latent_len)
        
        s_ce = ce_on_mask(s_logits, s_ids, s_ym)
        t_ce = ce_on_mask(t_logits, t_ids, t_ym)
        
        kl_student, kl_teacher = kl_balanced(t_z_log, s_z_log)
        
        sl = s_ce + (cfg.kl_balance_alpha * beta) * kl_student
        tl = t_ce + beta * kl_teacher
        
        # Делим лосс для grad_accum
        total_loss = (sl + tl) / cfg.grad_accum
        total_loss.backward()
        
        logger.update(sl.item(), s_ce.item(), kl_student.item(), t_ce.item(), kl_teacher.item(), time.time() - t0)
        
        # Делаем step только если накопили нужное количество градиентов
        if (step + 1) % cfg.grad_accum == 0:
            lr = get_lr(step)
            for opt in [opt_t, opt_s]:
                for g in opt.param_groups: g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
            opt_t.step(); opt_t.zero_grad()
            opt_s.step(); opt_s.zero_grad()
            
            # Синхронизация весов Учителя
            if (step // cfg.grad_accum) % cfg.sync_every_n_steps == 0:
                # Передаем веса в CPU, чтобы избежать утечек CUDA через очередь
                sd = {k: v.cpu() for k, v in teacher.state_dict().items() if "lora" in k}
                # Если очередь полна, заменяем старые веса новыми
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
            logger.log_sample(step, batch, z_list, s_logits, s_ym, tokenizer, beta)

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
