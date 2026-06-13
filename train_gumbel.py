"""
Latent Reasoning via ELBO — Qwen3.5-0.8B  v2
=============================================
Изменения vs v1:
  • latent_len = 128  (было 32)
  • beta_end   = 3.0  (было 1.0) — давит читаемость учителя
  • entropy bonus для учителя: −entropy_coef · H(q)
    стимулирует учителя генерировать менее предсказуемые z
  • Компактный прогресс-бар + подробный z-сэмпл каждые sample_every шагов

Архитектура:
  Учитель  q_φ : [Q, A, <think>, z₁…z₁₂₈]        — видит A до z
  Ученик   p_θ : [Q, <think>, z₁…z₁₂₈, </think>, A]  — не видит A до z

  L_student = CE(A|z,Q)  + β · KL(q‖p)
  L_teacher = CE(A|ctx) + β · KL(q‖p)  − ent_coef · H(q)
"""

import os
import sys, math, time, re
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════
#  Конфигурация
# ══════════════════════════════════════════════════════════════

@dataclass
class Config:
    model_name: str        = "Qwen/Qwen3.5-0.8B"
    dataset_name: str      = "AI-MO/NuminaMath-CoT"
    max_train_samples: int = 50_000
    max_q_tokens: int      = 192
    max_a_tokens: int      = 48
    latent_len: int        = 128          # ↑ было 32

    # ELBO Curriculum
    beta_start: float      = 0.0
    beta_peak: float       = 1.0          # Жёсткое давление на сжатие (пик)
    beta_final: float      = 0.01         # Финальная бета для работы на качество CE
    beta_warmup_steps: int = 400          # Фаза 1: 0..400 шагов (подъём до 1.0)
    beta_relax_steps: int  = 800          # Фаза 2: 400..800 шагов (спуск до 0.01)
    
    # Anti-Shortcut
    mi_target: float       = 12.0         # Целевой лосс для слепого Студента (как у человеческого текста)
    mi_coef: float         = 1.7          # Сила штрафа, если mi падает ниже mi_target
    
    # Энтропия
    ent_coef: float        = 0.5          # Коэффициент энтропии (ШТРАФ за высокую локальную энтропию)
    global_ent_coef: float = 0.5          # Бонус за глобальное разнообразие словаря (InfoMax)
    
    # Штраф за энтропию не нужен в Gumbel-Softmax, потому что токены всегда дискретные (hard=True)
    tau_start: float       = 2.0
    tau_end: float         = 0.5
    tau_anneal_steps: int  = 400

    # Обучение
    batch_size: int        = 1            # на 1 GPU
    grad_accum: int        = 16           # эфф. batch = 1 * 2(gpus) * 16 = 32
    lr: float              = 1e-4         # Увеличено для "широкой" LoRA
    max_steps: int         = 1000
    warmup_steps: int      = 100
    max_grad_norm: float   = 1.0

    # LoRA
    lora_r: int            = 128          # Огромная LoRA (~65M параметров)
    lora_alpha: int        = 256
    lora_dropout: float    = 0.05

    # Железо
    dtype: str             = "float16"
    teacher_device: str    = "cuda:0"
    student_device: str    = "cuda:0"     # cuda:1 если две GPU

    # Логирование
    log_every: int         = 10
    sample_every: int      = 100
    save_every: int        = 500
    output_dir: str        = "./checkpoints"
    use_wandb: bool        = False

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ══════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════

def get_beta(step: int) -> float:
    if step <= cfg.beta_warmup_steps:
        # Фаза 1: Подъём (Compression)
        t = step / max(1, cfg.beta_warmup_steps)
        return cfg.beta_start + (cfg.beta_peak - cfg.beta_start) * t
    elif step <= cfg.beta_relax_steps:
        # Фаза 2: Спуск (Quality Focus)
        t = (step - cfg.beta_warmup_steps) / max(1, cfg.beta_relax_steps - cfg.beta_warmup_steps)
        return cfg.beta_peak + (cfg.beta_final - cfg.beta_peak) * t
    else:
        # Фаза 3: Плато
        return cfg.beta_final

def get_tau(step: int) -> float:
    """Охлаждение Gumbel-Softmax от 2.0 до 0.5"""
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

class CrossDeviceCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dev):
        ctx.src = x.device; return x.to(dev)
    @staticmethod
    def backward(ctx, g):
        return g.to(ctx.src), None


# ══════════════════════════════════════════════════════════════
#  Датасет
# ══════════════════════════════════════════════════════════════

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
        print(f"  Датасет загружен: {len(self.samples):,} примеров")

    @staticmethod
    def _extract(item):
        if item.get("answer"):
            return str(item["answer"]).strip()
        boxes = re.findall(r"\\boxed\{([^}]+)\}", item.get("solution", ""))
        return boxes[-1].strip() if boxes else None

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ══════════════════════════════════════════════════════════════
#  Построение последовательностей
# ══════════════════════════════════════════════════════════════

def build_sequences(batch, tokenizer, z_list):
    """
    Учитель : [BOS, Q, A, <think>, z₁…zN]
    Ученик  : [BOS, Q, <think>, z₁…zN, </think>, \nA: , a]
    """
    def tok(t): return tokenizer(t, add_special_tokens=False)["input_ids"]

    TS = tok("<think>\n"); TE = tok("\n</think>\n"); AP = tok("\nA: ")
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []

    t_seqs, t_zm_log, t_ym = [], [], []
    s_seqs, s_zm_log, s_zm_embed, s_ym, s_qm = [], [], [], [], []

    for i, item in enumerate(batch):
        q_ids = tok(f"Q: {item['q']}\n")[:cfg.max_q_tokens]
        a_ids = tok(item["a"])[:cfg.max_a_tokens]
        z     = z_list[i]

        # ── Учитель ──
        t = bos + q_ids + a_ids + TS + z
        tz_s = len(bos) + len(q_ids) + len(a_ids) + len(TS)
        tz_e = tz_s + len(z)
        ty_s = len(bos) + len(q_ids)
        ty_e = ty_s + len(a_ids)
        t_seqs.append(t)
        t_zm_log.append(_mask(len(t), tz_s - 1, tz_e - 1))
        t_ym.append(_mask(len(t), ty_s, ty_e))

        # ── Ученик ──
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

    t_ids, t_at, (tzm_log, tym) = _pad(t_seqs, [t_zm_log, t_ym], cfg.teacher_device)
    s_ids, s_at, (szm_log, szm_embed, sym, sqm) = _pad(s_seqs, [s_zm_log, s_zm_embed, s_ym, s_qm], cfg.student_device)
    return (t_ids, t_at, tzm_log, tym), (s_ids, s_at, szm_log, szm_embed, sym, sqm)

def _mask(length, start, end):
    m = [False] * length
    for j in range(start, min(end, length)):
        m[j] = True
    return m


# ══════════════════════════════════════════════════════════════
#  Логиты по маске и Soft Embeddings
# ══════════════════════════════════════════════════════════════

def build_student_inputs_embeds(student, s_input_ids, s_z_mask, soft_z_embeds, device):
    embed_layer = student.get_input_embeddings()
    base_embeds = embed_layer(s_input_ids.to(device))
    z_mask_dev = s_z_mask.to(device)
    
    # Клонируем для сохранения графа и избежания изменения leaf-тензора (хотя base_embeds — результат функции)
    result = base_embeds.clone()
    
    # Маска s_z_mask имеет ровно cfg.latent_len True-значений для каждого батча.
    # soft_z_embeds имеет форму (B, L_z, D). Их можно просто заассайнить:
    result[z_mask_dev] = soft_z_embeds.reshape(-1, base_embeds.size(-1))
    
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
                chunk = torch.cat([chunk,
                    torch.zeros(Z-n, V, device=logits.device, dtype=logits.dtype)])
        out.append(chunk)
    return torch.stack(out)   # (B, Z, V)


# ══════════════════════════════════════════════════════════════
#  Loss-функции
# ══════════════════════════════════════════════════════════════

def ce_on_mask(logits, ids, mask):
    """Cross-entropy (СУММА по длине, СРЕДНЕЕ по батчу) - Оптимизированная версия"""
    B, L, V = logits.shape
    logits_shift = logits[:, :-1]
    targets = ids[:, 1:]
    ym = mask[:, 1:]
    
    valid_idx = ym.reshape(-1).nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    valid_logits = logits_shift.reshape(-1, V)[valid_idx]
    valid_targets = targets.reshape(-1)[valid_idx]
    
    # Считаем сумму CE только для целевых токенов
    sum_ce = F.cross_entropy(valid_logits.float(), valid_targets, reduction="sum")
    
    # Делим на размер батча B, чтобы получить (Сумма_на_последовательность).mean()
    return sum_ce / B

def kl_and_entropy(q_logits, p_logits):
    """СРЕДНЕЕ по батчу и СРЕДНЕЕ по длине (по просьбе: оставляем ослабленный штраф)"""
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    
    # KL(p||q) для каждого токена: (B, L)
    kl_per_token = F.kl_div(p_log, q_log, reduction='none', log_target=True).sum(-1)
    kl = kl_per_token.mean()
    
    # Локальная энтропия H_local(q) для каждого токена: (B, L)
    q = q_log.exp()
    # Защита от NaN: когда q -> 0, q_log -> -inf, и 0 * -inf даёт NaN.
    # Поэтому мы ограничиваем q_log снизу (например, -100).
    ent_per_token = -(q * q_log.clamp(min=-100)).sum(-1)
    ent_local = ent_per_token.mean()
    
    # Глобальная энтропия H_global (InfoMax)
    q_global = q.mean(dim=1) # (B, V) усредняем вероятности по длине
    ent_global = -(q_global * (q_global + 1e-9).log()).sum(-1).mean()
    
    return kl, ent_local, ent_global

def student_loss(t_z, s_z, s_all, s_ids, s_ym, beta):
    """L_s = CE(y) + β·KL"""
    ce  = ce_on_mask(s_all, s_ids, s_ym)
    kl, _, _ = kl_and_entropy(t_z.detach(), s_z)
    return ce + beta * kl, ce.item(), kl.item()

def teacher_loss_fn(t_all, t_z, s_z, t_ids, t_ym, beta, ent_coef, global_ent_coef):
    """L_t = CE(y) + β·KL + ent_coef·H_local - global_ent_coef·H_global"""
    ce       = ce_on_mask(t_all, t_ids, t_ym)
    kl, ent_local, ent_global  = kl_and_entropy(t_z, s_z.detach())
    
    # ПРИБАВЛЯЕМ локальную энтропию (ШТРАФ за неуверенность)
    # ВЫЧИТАЕМ глобальную энтропию (БОНУС за богатство словаря)
    total_loss = ce + beta * kl + ent_coef * ent_local - global_ent_coef * ent_global
    
    return total_loss, ce.item(), kl.item(), ent_local.item(), ent_global.item()


# ══════════════════════════════════════════════════════════════
#  Логирование
# ══════════════════════════════════════════════════════════════

class Logger:
    """Накапливает метрики и печатает компактную строку + красивый z-сэмпл."""
    def __init__(self):
        self.reset()
        self._t0 = time.time()
        self._step_times = []

    def reset(self):
        if hasattr(self, '_n') and self._n > 0:
            self.last_logged_ent = self.s.get("t_ent", 0.0) / self._n
        elif not hasattr(self, 'last_logged_ent'):
            self.last_logged_ent = 0.0
            
        self.s = dict(loss=0., s_ce=0., s_kl=0., t_ce=0., t_kl=0., t_ent=0., t_ent_g=0., mi_ce=0.)
        self._n = 0

    def update(self, loss, s_ce, s_kl, t_ce, t_kl, t_ent, t_ent_g, mi_ce, step_t):
        self.s["loss"]  += loss
        self.s["s_ce"]  += s_ce
        self.s["s_kl"]  += s_kl
        self.s["t_ce"]  += t_ce
        self.s["t_kl"]  += t_kl
        self.s["t_ent"] += t_ent
        self.s["t_ent_g"] += t_ent_g
        self.s["mi_ce"] += mi_ce
        self._n         += 1
        self._step_times.append(step_t)

    def log_step(self, step, beta, lr, mem_t, mem_s):
        n = self._n or 1
        avg_t = sum(self._step_times[-20:]) / len(self._step_times[-20:])
        eta   = avg_t * (cfg.max_steps - step)
        elapsed = time.time() - self._t0

        bar = fmt_bar(step, cfg.max_steps)
        pct = 100 * step / cfg.max_steps

        print(
            f"\r[{bar}] {pct:4.1f}%  step={step}/{cfg.max_steps}"
            f"  loss={self.s['loss']/n:.3f}"
            f"  s_ce={self.s['s_ce']/n:.3f}  s_kl={self.s['s_kl']/n:.3f}"
            f"  t_ce={self.s['t_ce']/n:.3f}  t_kl={self.s['t_kl']/n:.3f}"
            f"  H_L={self.s['t_ent']/n:.3f}  H_G={self.s['t_ent_g']/n:.3f}  mi={self.s['mi_ce']/n:.2f}"
            f"  β={beta:.2f}  lr={lr:.1e}"
            f"  {avg_t:.2f}s/it  ⏱{fmt_time(elapsed)}  ETA={fmt_time(eta)}"
            f"  VRAM=[{mem_t:.1f}|{mem_s:.1f}]G",
            flush=True
        )
        self.reset()

    def log_sample(self, step, batch, z_list, s_logits, s_ym, tokenizer, beta):
        """Красивый z-сэмпл — показывает как учитель эволюционирует."""
        z0     = z_list[0]
        z_text = tokenizer.decode(z0, skip_special_tokens=False)
        uniq   = len(set(z0))
        top    = Counter(z0).most_common(8)

        # Предсказание ученика
        y_idx = s_ym[0].nonzero(as_tuple=True)[0]
        pred_a = ""
        if len(y_idx) > 0:
            pred_ids = s_logits[0, y_idx[0]-1:y_idx[-1]].argmax(-1)
            pred_a   = tokenizer.decode(pred_ids.cpu(), skip_special_tokens=True)

        # Читаемость z: доля токенов которые декодируются в обычные слова
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

        if cfg.use_wandb:
            import wandb
            wandb.log({"readability": readability,
                       "z_unique_ratio": uniq / cfg.latent_len,
                       "step": step})

    def wandb_log(self, step, beta, lr):
        if not cfg.use_wandb: return
        import wandb
        n = self._n or 1
        wandb.log({k: v/n for k, v in self.s.items()} |
                  {"beta": beta, "lr": lr, "step": step})


# ══════════════════════════════════════════════════════════════
#  Главный цикл
# ══════════════════════════════════════════════════════════════

def train():
    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum)
    cfg.teacher_device = accelerator.device
    cfg.student_device = accelerator.device

    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    if accelerator.is_main_process:
        print("═══════════════════════════════════════════════════════════════════")
        print(f"  Latent Reasoning ELBO  ·  {cfg.model_name}")
        print(f"  latent_len={cfg.latent_len}  β: {cfg.beta_start} → {cfg.beta_peak} → {cfg.beta_final}")
        print("═══════════════════════════════════════════════════════════════════")

    print("📦 Токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    if accelerator.is_main_process: print(f"📦 Загрузка моделей...")
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype,
        device_map={"": accelerator.device}, trust_remote_code=True,
        attn_implementation="sdpa")
    teacher = get_peft_model(teacher, lora_cfg)
    teacher.gradient_checkpointing_enable()
    teacher.train(); teacher.config.use_cache = False

    student = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype,
        device_map={"": accelerator.device}, trust_remote_code=True,
        attn_implementation="sdpa")
    student = get_peft_model(student, lora_cfg)
    student.gradient_checkpointing_enable()
    student.train(); student.config.use_cache = False

    import bitsandbytes as bnb
    opt_t = bnb.optim.PagedAdamW8bit(teacher.parameters(), lr=cfg.lr, weight_decay=0.01)
    opt_s = bnb.optim.PagedAdamW8bit(student.parameters(), lr=cfg.lr, weight_decay=0.01)

    if accelerator.is_main_process: print("📦 Датасет...")
    ds = MathQADataset(tokenizer, max_samples=cfg.max_train_samples)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, collate_fn=lambda x: x)

    teacher, student, opt_t, opt_s, loader = accelerator.prepare(teacher, student, opt_t, opt_s, loader)

    if accelerator.is_main_process and cfg.use_wandb:
        import wandb; wandb.init(project="latent-elbo-qwen35", config=vars(cfg))

    logger = Logger()
    step = 0
    lr   = cfg.lr

    print(f"\n🚀 Старт обучения ({cfg.max_steps} шагов)...\n")

    for _ in range(9999):
        for batch in loader:
            if step >= cfg.max_steps: break
            t0   = time.time()
            beta = get_beta(step)
            tau = get_tau(step)

            # ── 1. Генерация z учителем ──
            teacher.eval(); accelerator.unwrap_model(teacher).config.use_cache = True
            prompts = []
            for item in batch:
                text = f"Q: {item['q'][:400]}\nA: {item['a'][:80]}\n<think>\n"
                ids  = tokenizer(text, add_special_tokens=True,
                                 return_tensors="pt")["input_ids"][0].tolist()
                prompts.append(ids[:cfg.max_q_tokens + cfg.max_a_tokens + 10])

            mp   = max(len(p) for p in prompts)
            pad  = tokenizer.pad_token_id
            B    = len(batch)
            gids = torch.full((B, mp), pad, dtype=torch.long, device=cfg.teacher_device)
            gmsk = torch.zeros(B, mp, dtype=torch.long, device=cfg.teacher_device)
            for i, p in enumerate(prompts):
                gids[i, mp-len(p):] = torch.tensor(p, dtype=torch.long)
                gmsk[i, mp-len(p):] = 1

            # Adaptive Temperature Annealing for Generation
            # Динамическая температура для генерации примеров (от 0.3 до 0.9 в зависимости от H_local)
            gen_temp = max(0.3, 0.9 - max(0.0, (logger.last_logged_ent - 10.0)) * 0.05)
            
            with torch.no_grad():
                gen = accelerator.unwrap_model(teacher).generate(
                    input_ids=gids, attention_mask=gmsk,
                    max_new_tokens=cfg.latent_len, min_new_tokens=cfg.latent_len,
                    do_sample=True, temperature=gen_temp, top_p=0.9,
                    pad_token_id=pad, eos_token_id=None, use_cache=True)

            z_list = []
            for i in range(B):
                z = gen[i, mp:mp+cfg.latent_len].cpu().tolist()
                if len(z) < cfg.latent_len:
                    z += [pad] * (cfg.latent_len - len(z))
                z_list.append(z[:cfg.latent_len])

            accelerator.unwrap_model(teacher).config.use_cache = False; teacher.train()

            # ── 2. Строим последовательности ──
            (t_ids, t_at, t_zm_log, t_ym), \
            (s_ids, s_at, s_zm_log, s_zm_embed, s_ym, s_qm) = build_sequences(batch, tokenizer, z_list)

            with accelerator.accumulate(teacher, student):
                # ── 3. Forward учителя ──
                t_out    = teacher(input_ids=t_ids, attention_mask=t_at)
                t_logits = t_out.logits
                t_z_log  = masked_logits(t_logits, t_zm_log, cfg.latent_len)

                # ── 4. Forward ученика (Gumbel-Softmax Embeddings) ──
                s_embed_matrix = accelerator.unwrap_model(student).get_input_embeddings().weight
                t_z_probs = F.gumbel_softmax(t_z_log.float(), tau=tau, hard=True, dim=-1).to(s_embed_matrix.dtype)
                s_embed_on_teacher = s_embed_matrix.detach().to(cfg.teacher_device)
                soft_z_embeds = torch.matmul(t_z_probs, s_embed_on_teacher)
                
                s_inputs_embeds = build_student_inputs_embeds(
                    accelerator.unwrap_model(student), s_ids, s_zm_embed,
                    CrossDeviceCopy.apply(soft_z_embeds, cfg.student_device), cfg.student_device
                )
                s_out    = student(inputs_embeds=s_inputs_embeds, attention_mask=s_at)
                s_logits = s_out.logits
                s_z_log  = masked_logits(s_logits, s_zm_log, cfg.latent_len)

                # ── 4. Anti-Shortcut Penalty (Студент без Q) ──
                s_at_no_q = s_at.clone()
                s_at_no_q[s_qm] = 0  # маскируем Q в attention
                
                # Отделяем граф вычислений от s_inputs_embeds, чтобы безопасно 
                # использовать retain_graph=False и освобождать VRAM без крашей.
                s_inputs_embeds_no_q = s_inputs_embeds.detach().clone()
                s_inputs_embeds_no_q[s_qm] = 0.0  # зануляем Q
                s_z_embeds_dev = CrossDeviceCopy.apply(soft_z_embeds, cfg.student_device)
                s_inputs_embeds_no_q[s_zm_embed] = s_z_embeds_dev.reshape(-1, s_inputs_embeds.size(-1))

                # Создаём правильные position_ids ДО маскирования Q (фикс бага с RoPE)
                s_position_ids = (s_at.cumsum(dim=-1) - 1).clamp(min=0)
                
                s_out_no_q = student(inputs_embeds=s_inputs_embeds_no_q, attention_mask=s_at_no_q, position_ids=s_position_ids)
                ce_z_only = ce_on_mask(s_out_no_q.logits, s_ids, s_ym)
                # Экспоненциальный штраф-стена: если mi падает ниже mi_target (12.0),
                # градиент взрывается по экспоненте, полностью подавляя CE лосс.
                # НО мы ограничиваем diff до 3.0 (штраф ~32), чтобы избежать 
                # экстремального клиппинга градиентов, который замораживает Учителя.
                diff = F.relu(cfg.mi_target - ce_z_only)
                diff = torch.clamp(diff, max=3.0)
                anti_shortcut_loss = cfg.mi_coef * (torch.exp(diff) - 1.0)
                
                # Пропускаем градиент штрафа только в soft_z_embeds (чтобы не портить веса Студента)
                # retain_graph=False освобождает граф вычислений слепого Студента, решая утечку VRAM
                grad_z = torch.autograd.grad(anti_shortcut_loss, soft_z_embeds, retain_graph=False)[0]
                
                # Создаем суррогатный лосс для единого backward (решает проблему крэша DDP)
                surrogate_loss = (soft_z_embeds * grad_z.detach()).sum()

                # ── 5. Loss ──
                t_z_on_s = CrossDeviceCopy.apply(t_z_log, cfg.student_device)
                s_z_on_t = CrossDeviceCopy.apply(s_z_log, cfg.teacher_device)

                sl, s_ce, s_kl = student_loss(
                    t_z_on_s, s_z_log, s_logits, s_ids, s_ym, beta)

                tl, t_ce, t_kl, t_ent, t_ent_g = teacher_loss_fn(
                    t_logits, t_z_log, s_z_on_t.detach(),
                    t_ids, t_ym, beta, cfg.ent_coef, cfg.global_ent_coef)

                # ── 6. Оптимизация: совместный backward для экономии памяти ──
                total_loss = sl + tl.to(sl.device) + surrogate_loss.to(sl.device)
                accelerator.backward(total_loss)
                logger.update(sl.item(), s_ce, s_kl, t_ce, t_kl, t_ent, t_ent_g, ce_z_only.item(),
                              time.time() - t0)

                # ── 6. Optimizer step ──
                lr = get_lr(step)
                for opt, model in [(opt_t, teacher), (opt_s, student)]:
                    for g in opt.param_groups: g["lr"] = lr
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step(); opt.zero_grad()

            # ── 7. Логирование ──
            if accelerator.is_main_process:
                if step % cfg.log_every == 0 and step > 0:
                    mt = torch.cuda.max_memory_allocated(cfg.teacher_device) / 1e9
                    ms = torch.cuda.max_memory_allocated(cfg.student_device) / 1e9
                    logger.log_step(step, beta, lr, mt, ms)
                    logger.wandb_log(step, beta, lr)

                if step % cfg.sample_every == 0:
                    logger.log_sample(step, batch, z_list,
                                      s_logits, s_ym, tokenizer, beta)

                if step % cfg.save_every == 0 and step > 0:
                    ckpt = f"{cfg.output_dir}/step_{step}"
                    os.makedirs(ckpt, exist_ok=True)
                    accelerator.unwrap_model(student).save_pretrained(f"{ckpt}/student")
                    accelerator.unwrap_model(teacher).save_pretrained(f"{ckpt}/teacher")
                    tokenizer.save_pretrained(ckpt)
                    print(f"\n  💾 Чекпоинт → {ckpt}\n")

            step += 1

        if step >= cfg.max_steps: break

    if accelerator.is_main_process:
        final = f"{cfg.output_dir}/final"
        os.makedirs(final, exist_ok=True)
        accelerator.unwrap_model(student).save_pretrained(f"{final}/student")
        accelerator.unwrap_model(teacher).save_pretrained(f"{final}/teacher")
        tokenizer.save_pretrained(final)
        print(f"\n\n✅ Готово. Чекпоинт → {final}")


if __name__ == "__main__":
    import os, sys
    if "LOCAL_RANK" not in os.environ:
        os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
        os.system("pip install bitsandbytes accelerate datasets git+https://github.com/huggingface/transformers.git wandb peft")

        import torch
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            cap = torch.cuda.get_device_capability(i)
            name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {name}  capability={cap[0]}.{cap[1]}")

        if num_gpus == 0:
            print("❌ No GPUs found!")
            sys.exit(1)

        gpu_ids = [str(i) for i in range(num_gpus)]
        print(f"\n🚀 Launching on {num_gpus} GPU(s): [{', '.join(gpu_ids)}]")

        if num_gpus == 1:
            dist_type = "NO"
        else:
            dist_type = "MULTI_GPU"

        config = f"""compute_environment: LOCAL_MACHINE
distributed_type: {dist_type}
downcast_bf16: 'no'
gpu_ids: '{','.join(gpu_ids)}'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: {num_gpus}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false"""
        with open("default_config.yaml", "w") as f: f.write(config)
        ret = os.system(f"accelerate launch --config_file default_config.yaml {sys.argv[0]}")
        sys.exit(ret >> 8)
    else:
        train()