"""
Latent Reasoning via ELBO — teacher/student training.

Схема:
  - Учитель q_φ(z|x,y): получает [x, y, <latent>, z₁…zₙ] — y СЛЕВА от z,
    causal attention позволяет видеть y при генерации каждого zₜ
  - Ученик p_θ(z|x):    получает [x, <latent>, z₁…zₙ, y] — y СПРАВА от z,
    causal mask гарантирует, что z не видит y
  - ELBO = CE(y | z, x) − β · Σ_t KL(q_t ‖ p_t)
  - KL обучает ОБОИХ: учитель регуляризуется (не уходить от ученика),
    ученик дистиллируется (приближаться к учителю)
  - z — обычные токены из словаря, нет штрафа за нечитаемость
  - Архитектура не меняется, учитель и ученик различаются только порядком y и z
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dataclasses import dataclass
import math
import time
import os


# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────

@dataclass
class Config:
    # Модель
    model_name: str = "google/gemma-2-2b-it"

    # Данные
    dataset_name: str = "AI-MO/NuminaMath-CoT"
    max_train_samples: int = 50_000
    max_question_len: int = 192
    max_answer_len: int = 64
    latent_len: int = 32

    # ELBO
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_steps: int = 1500

    # Обучение
    batch_size: int = 2
    grad_accum: int = 16              # эффективный batch = 32
    lr_teacher: float = 1e-5
    lr_student: float = 1e-5
    max_steps: int = 2000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Железо
    dtype: str = "float16"
    teacher_device: str = "cuda:0"
    student_device: str = "cuda:1"

    # Оптимизации памяти
    use_grad_checkpoint: bool = True
    use_8bit_optim: bool = True

    # Логирование
    log_every: int = 10
    sample_every: int = 100           # показать z-токены
    save_every: int = 500
    output_dir: str = "./checkpoints"
    use_wandb: bool = False


cfg = Config()

LATENT_START = "<latent>"
LATENT_END   = "</latent>"
LATENT_PAD   = "▁"


# ─────────────────────────────────────────────
# Датасет
# ─────────────────────────────────────────────

class MathQADataset(Dataset):
    def __init__(self, tokenizer, split="train", max_samples=50_000):
        raw = load_dataset(cfg.dataset_name, split=split, streaming=False)
        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.tokenizer = tokenizer
        self.samples = []

        for item in raw:
            problem = item.get("problem", "")
            answer = self._extract_answer(item)
            if problem and answer:
                self.samples.append({"question": problem, "answer": answer})

    def _extract_answer(self, item):
        if "answer" in item and item["answer"]:
            return str(item["answer"])
        sol = item.get("solution", "")
        import re
        boxes = re.findall(r"\\boxed\{([^}]+)\}", sol)
        return boxes[-1] if boxes else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# Collate: раздельные последовательности
# ─────────────────────────────────────────────

def collate_fn(batch, tokenizer, latent_len):
    """
    teacher: [x, y, <latent>, PAD*K, </latent>]  — y СЛЕВА от z
    student: [x, <latent>, PAD*K, </latent>, y]   — y СПРАВА от z
    """
    questions = [item["question"] for item in batch]
    answers   = [item["answer"]   for item in batch]

    latent_placeholder = LATENT_PAD * latent_len
    max_len = cfg.max_question_len + latent_len + cfg.max_answer_len + 32

    teacher_texts = [
        f"Question: {q}\nAnswer: {a}\n{LATENT_START}{latent_placeholder}{LATENT_END}"
        for q, a in zip(questions, answers)
    ]
    student_texts = [
        f"Question: {q}\n{LATENT_START}{latent_placeholder}{LATENT_END}\nAnswer: {a}"
        for q, a in zip(questions, answers)
    ]

    teacher_enc = tokenizer(teacher_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
    student_enc = tokenizer(student_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)

    latent_start_id = tokenizer.encode(LATENT_START, add_special_tokens=False)[0]
    latent_end_id   = tokenizer.encode(LATENT_END,   add_special_tokens=False)[0]

    q_prefix_lens = []
    for q in questions:
        prefix_ids = tokenizer.encode(f"Question: {q}\n", add_special_tokens=True)
        q_prefix_lens.append(len(prefix_ids))

    t_ids, t_attn = teacher_enc["input_ids"], teacher_enc["attention_mask"]
    s_ids, s_attn = student_enc["input_ids"], student_enc["attention_mask"]
    B, L_t = t_ids.shape
    L_s = s_ids.shape[1]

    teacher_z_mask = torch.zeros(B, L_t, dtype=torch.bool)
    teacher_y_mask = torch.zeros(B, L_t, dtype=torch.bool)
    student_z_mask = torch.zeros(B, L_s, dtype=torch.bool)
    student_y_mask = torch.zeros(B, L_s, dtype=torch.bool)

    for i in range(B):
        t_list = t_ids[i].tolist()
        try:
            t_ls = t_list.index(latent_start_id)
            t_le = t_list.index(latent_end_id)
            teacher_z_mask[i, t_ls + 1 : t_le] = True
            teacher_y_mask[i, q_prefix_lens[i] : t_ls] = True
        except ValueError:
            pass

        s_list = s_ids[i].tolist()
        try:
            s_ls = s_list.index(latent_start_id)
            s_le = s_list.index(latent_end_id)
            student_z_mask[i, s_ls + 1 : s_le] = True
            student_y_mask[i, s_le + 1 :] = True
            student_y_mask[i] &= s_attn[i].bool()
        except ValueError:
            pass

    return {
        "teacher_input_ids": t_ids, "teacher_attention_mask": t_attn,
        "teacher_z_mask": teacher_z_mask, "teacher_y_mask": teacher_y_mask,
        "student_input_ids": s_ids, "student_attention_mask": s_attn,
        "student_z_mask": student_z_mask, "student_y_mask": student_y_mask,
        "questions": questions, "answers": answers,
    }


# ─────────────────────────────────────────────
# Хелперы
# ─────────────────────────────────────────────

def extract_masked_logits(logits, mask, expected_len):
    """Извлекает логиты по маске, сохраняя autograd."""
    B, L, V = logits.shape
    slices = []
    for i in range(B):
        indices = mask[i].nonzero(as_tuple=True)[0]
        n = min(len(indices), expected_len)
        extracted = logits[i, indices[:n]]
        if n < expected_len:
            pad = torch.zeros(expected_len - n, V, device=logits.device, dtype=logits.dtype)
            extracted = torch.cat([extracted, pad], dim=0)
        slices.append(extracted)
    return torch.stack(slices, dim=0)


def fill_z_tokens(input_ids, z_mask, z_tokens, latent_len):
    """Заменяет PAD на z-позициях реальными токенами учителя."""
    result = input_ids.clone()
    B = input_ids.shape[0]
    for i in range(B):
        indices = z_mask[i].nonzero(as_tuple=True)[0]
        n = min(len(indices), latent_len)
        result[i, indices[:n]] = z_tokens[i, :n]
    return result


def compute_kl(q_logits, p_logits):
    """KL(q ‖ p), усреднённый по позициям."""
    q_probs = F.softmax(q_logits, dim=-1)
    p_log_p = F.log_softmax(p_logits, dim=-1)
    kl = (q_probs * (q_probs.clamp(min=1e-9).log() - p_log_p)).sum(dim=-1)
    return kl.mean()


# ─────────────────────────────────────────────
# ELBO loss (student side)
# ─────────────────────────────────────────────

def compute_elbo_loss(teacher_z_logits, student_z_logits, student_logits,
                      student_input_ids, student_y_mask, beta):
    B, L_s, V = student_logits.shape

    # Реконструкция y
    student_log_probs = F.log_softmax(student_logits[:, :-1, :], dim=-1)
    targets        = student_input_ids[:, 1:]
    y_mask_shifted = student_y_mask[:, 1:]

    ce_all = F.nll_loss(
        student_log_probs.reshape(-1, V), targets.reshape(-1),
        reduction="none",
    ).reshape(B, L_s - 1)

    n_y = y_mask_shifted.float().sum().clamp(min=1)
    loss_recon = (ce_all * y_mask_shifted.float()).sum() / n_y

    # KL (student side, teacher detached)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_z_logits.detach(), dim=-1)
    student_log_p = F.log_softmax(student_z_logits, dim=-1)
    kl_per_pos = (teacher_probs * (teacher_probs.clamp(min=1e-9).log() - student_log_p)).sum(dim=-1)
    loss_kl = kl_per_pos.mean()

    loss = loss_recon + beta * loss_kl
    return loss, loss_recon.item(), loss_kl.item()


def get_beta(step):
    if step >= cfg.beta_warmup_steps:
        return cfg.beta_end
    return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * (step / cfg.beta_warmup_steps)


def get_lr(step, base_lr):
    if step < cfg.warmup_steps:
        return base_lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────
# Форматирование времени
# ─────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


# ─────────────────────────────────────────────
# Тренировочный цикл
# ─────────────────────────────────────────────

def train():
    torch_dtype = torch.float16 if cfg.dtype == "float16" else torch.bfloat16

    print("=" * 70)
    print("  LATENT REASONING — ELBO TRAINING")
    print("=" * 70)
    print(f"  Model:      {cfg.model_name}")
    print(f"  Dtype:      {cfg.dtype}")
    print(f"  Devices:    teacher={cfg.teacher_device}  student={cfg.student_device}")
    print(f"  Batch:      {cfg.batch_size} × {cfg.grad_accum} = {cfg.batch_size * cfg.grad_accum}")
    print(f"  Latent len: {cfg.latent_len}")
    print(f"  Steps:      {cfg.max_steps}")
    print(f"  8bit optim: {cfg.use_8bit_optim}")
    print(f"  Grad ckpt:  {cfg.use_grad_checkpoint}")
    print("=" * 70)

    # ── Токенизатор ──
    print("\n📦 Загружаем токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": [LATENT_START, LATENT_END]}
    tokenizer.add_special_tokens(special_tokens)

    # ── Модели ──
    print(f"📦 Загружаем учителя → {cfg.teacher_device}...")
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch_dtype, device_map=cfg.teacher_device,
    )
    teacher.resize_token_embeddings(len(tokenizer))
    if cfg.use_grad_checkpoint:
        teacher.gradient_checkpointing_enable()
    teacher.train()

    print(f"📦 Загружаем ученика → {cfg.student_device}...")
    student = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch_dtype, device_map=cfg.student_device,
    )
    student.resize_token_embeddings(len(tokenizer))
    if cfg.use_grad_checkpoint:
        student.gradient_checkpointing_enable()
    student.train()

    # ── Оптимизаторы ──
    if cfg.use_8bit_optim:
        import bitsandbytes as bnb
        print("⚡ 8-bit AdamW (bitsandbytes)")
        opt_teacher = bnb.optim.AdamW8bit(teacher.parameters(), lr=cfg.lr_teacher, weight_decay=0.01)
        opt_student = bnb.optim.AdamW8bit(student.parameters(), lr=cfg.lr_student, weight_decay=0.01)
    else:
        opt_teacher = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr_teacher, weight_decay=0.01)
        opt_student = torch.optim.AdamW(student.parameters(), lr=cfg.lr_student, weight_decay=0.01)

    # ── Данные ──
    print("📦 Загружаем датасет...")
    dataset = MathQADataset(tokenizer, split="train", max_samples=cfg.max_train_samples)
    print(f"   → {len(dataset)} примеров")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg.latent_len),
        num_workers=2, pin_memory=True,
    )

    # ── VRAM после загрузки ──
    for dev in [cfg.teacher_device, cfg.student_device]:
        mem = torch.cuda.max_memory_allocated(dev) / 1e9
        print(f"   VRAM {dev}: {mem:.1f} GB")

    os.makedirs(cfg.output_dir, exist_ok=True)

    step = 0
    accum_loss = accum_recon = accum_kl = accum_tce = accum_tkl = 0.0
    lr = cfg.lr_student
    step_times = []
    train_start = time.time()

    print(f"\n🚀 Старт обучения ({cfg.max_steps} шагов)...\n")

    for epoch in range(9999):
        for batch in loader:
            if step >= cfg.max_steps:
                break

            step_t0 = time.time()

            t_input_ids = batch["teacher_input_ids"]
            t_attn_mask = batch["teacher_attention_mask"]
            t_z_mask    = batch["teacher_z_mask"]
            t_y_mask    = batch["teacher_y_mask"]
            s_input_ids = batch["student_input_ids"]
            s_attn_mask = batch["student_attention_mask"]
            s_z_mask    = batch["student_z_mask"]
            s_y_mask    = batch["student_y_mask"]

            beta = get_beta(step)

            # ── 1) Forward учителя ──
            t_ids  = t_input_ids.to(cfg.teacher_device)
            t_mask = t_attn_mask.to(cfg.teacher_device)
            teacher_out = teacher(input_ids=t_ids, attention_mask=t_mask)
            teacher_logits = teacher_out.logits

            teacher_z_logits = extract_masked_logits(
                teacher_logits, t_z_mask.to(cfg.teacher_device), cfg.latent_len
            )
            with torch.no_grad():
                z_tokens_from_teacher = teacher_z_logits.argmax(dim=-1)

            # ── 2) Forward ученика (с z-токенами учителя) ──
            s_ids_filled = fill_z_tokens(
                s_input_ids, s_z_mask,
                z_tokens_from_teacher.cpu(), cfg.latent_len
            )
            s_ids  = s_ids_filled.to(cfg.student_device)
            s_mask = s_attn_mask.to(cfg.student_device)
            student_out = student(input_ids=s_ids, attention_mask=s_mask)
            student_logits = student_out.logits

            student_z_logits = extract_masked_logits(
                student_logits, s_z_mask.to(cfg.student_device), cfg.latent_len
            )

            # ── 3) Student loss ──
            teacher_z_logits_s = teacher_z_logits.to(cfg.student_device).detach()
            loss, l_recon, l_kl = compute_elbo_loss(
                teacher_z_logits_s, student_z_logits, student_logits,
                s_ids, s_y_mask.to(cfg.student_device), beta,
            )

            # ── 4) Teacher loss = CE(y|x) + β·KL(teacher ‖ student_detached) ──
            t_log_p     = F.log_softmax(teacher_logits[:, :-1, :], dim=-1)
            t_targets   = t_ids[:, 1:]
            t_y_shifted = t_y_mask.to(cfg.teacher_device)[:, 1:]
            t_ce = F.nll_loss(
                t_log_p.reshape(-1, t_log_p.size(-1)),
                t_targets.reshape(-1), reduction="none",
            ).reshape(t_ids.size(0), -1)
            n_y_t = t_y_shifted.float().sum().clamp(min=1)
            teacher_ce = (t_ce * t_y_shifted.float()).sum() / n_y_t

            student_z_logits_t = student_z_logits.to(cfg.teacher_device).detach()
            teacher_kl = compute_kl(teacher_z_logits, student_z_logits_t)
            teacher_loss = teacher_ce + beta * teacher_kl

            # ── 5) Backward ──
            (loss / cfg.grad_accum).backward()
            (teacher_loss / cfg.grad_accum).backward()

            accum_loss  += loss.item()
            accum_recon += l_recon
            accum_kl    += l_kl
            accum_tce   += teacher_ce.item()
            accum_tkl   += teacher_kl.item()

            if (step + 1) % cfg.grad_accum == 0:
                lr = get_lr(step, cfg.lr_student)
                for g in opt_student.param_groups:
                    g["lr"] = lr
                for g in opt_teacher.param_groups:
                    g["lr"] = get_lr(step, cfg.lr_teacher)
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), cfg.max_grad_norm)
                opt_student.step(); opt_student.zero_grad()
                opt_teacher.step(); opt_teacher.zero_grad()

            step_time = time.time() - step_t0
            step_times.append(step_time)

            # ── Логирование ──
            if step % cfg.log_every == 0 and step > 0:
                n = cfg.log_every
                avg_t = sum(step_times[-n:]) / len(step_times[-n:])
                eta = avg_t * (cfg.max_steps - step)
                elapsed = time.time() - train_start
                mem_t = torch.cuda.max_memory_allocated(cfg.teacher_device) / 1e9
                mem_s = torch.cuda.max_memory_allocated(cfg.student_device) / 1e9

                print(
                    f"[{step:5d}/{cfg.max_steps}] "
                    f"loss={accum_loss/n:.4f}  "
                    f"recon={accum_recon/n:.4f}  "
                    f"kl={accum_kl/n:.4f}  "
                    f"t_ce={accum_tce/n:.4f}  "
                    f"t_kl={accum_tkl/n:.4f}  "
                    f"β={beta:.3f}  lr={lr:.1e}  "
                    f"{avg_t:.2f}s/step  "
                    f"⏱{fmt_time(elapsed)}  ETA {fmt_time(eta)}  "
                    f"VRAM[{mem_t:.1f}G|{mem_s:.1f}G]"
                )
                accum_loss = accum_recon = accum_kl = accum_tce = accum_tkl = 0.0

            # ── Сэмплы z-токенов ──
            if step % cfg.sample_every == 0:
                with torch.no_grad():
                    z_ids = z_tokens_from_teacher[0].cpu().tolist()
                    z_decoded = tokenizer.decode(z_ids, skip_special_tokens=False)
                    unique_z = len(set(z_ids))
                    total_z  = len(z_ids)

                    # Предсказание y учеником
                    y_indices = s_y_mask[0].nonzero(as_tuple=True)[0]
                    pred_y = ""
                    if len(y_indices) > 0:
                        pred_ids = student_logits[0, y_indices[0]-1 : y_indices[-1]].argmax(dim=-1)
                        pred_y = tokenizer.decode(pred_ids.cpu().tolist(), skip_special_tokens=True)

                    q_short = batch["questions"][0][:100]
                    a_short = batch["answers"][0][:60]

                print(f"\n{'─'*70}")
                print(f"  🔬 Z-SAMPLE  step={step}  unique={unique_z}/{total_z}")
                print(f"  Q: {q_short}")
                print(f"  A (true):   {a_short}")
                print(f"  A (student): {pred_y[:80]}")
                print(f"  z (decoded): {z_decoded[:250]}")
                # Топ-10 самых частых z-токенов
                from collections import Counter
                top = Counter(z_ids).most_common(8)
                top_str = "  ".join(
                    f"'{tokenizer.decode([tid])}'×{cnt}" for tid, cnt in top
                )
                print(f"  z top tokens: {top_str}")
                print(f"{'─'*70}\n")

            # ── Чекпоинт ──
            if step % cfg.save_every == 0 and step > 0:
                ckpt = f"{cfg.output_dir}/step_{step}"
                os.makedirs(ckpt, exist_ok=True)
                student.save_pretrained(f"{ckpt}/student")
                teacher.save_pretrained(f"{ckpt}/teacher")
                tokenizer.save_pretrained(ckpt)
                print(f"  💾 Чекпоинт → {ckpt}")

            step += 1

        if step >= cfg.max_steps:
            break

    # ── Финальное сохранение ──
    elapsed = time.time() - train_start
    print(f"\n{'='*70}")
    print(f"  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"  Шагов: {step}   Время: {fmt_time(elapsed)}")
    print(f"{'='*70}")

    final_dir = f"{cfg.output_dir}/final"
    os.makedirs(final_dir, exist_ok=True)
    student.save_pretrained(f"{final_dir}/student")
    teacher.save_pretrained(f"{final_dir}/teacher")
    tokenizer.save_pretrained(final_dir)
    print(f"  💾 Финальный чекпоинт → {final_dir}")


if __name__ == "__main__":
    train()
