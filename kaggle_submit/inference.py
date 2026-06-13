"""
Инференс: только ученик.
Генерирует z (латентные токены рассуждения), потом ответ y.
Учитель не нужен.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

LATENT_START = "<latent>"
LATENT_END   = "</latent>"


def load_student(checkpoint_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        f"{checkpoint_dir}/student",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_answer(
    tokenizer, model, question: str,
    latent_len: int = 32, max_answer_tokens: int = 128,
    temperature: float = 0.7,
) -> dict:
    device = next(model.parameters()).device

    prefix = f"Question: {question}\n{LATENT_START}"
    prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)

    # Шаг 1: генерируем z
    z_ids = model.generate(
        prefix_ids,
        max_new_tokens=latent_len,
        min_new_tokens=latent_len,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None,
    )
    z_tokens = z_ids[0, prefix_ids.shape[1]:]

    # Шаг 2: генерируем ответ
    latent_end_ids = tokenizer.encode(
        f"{LATENT_END}\nAnswer:", add_special_tokens=False, return_tensors="pt"
    ).to(device)
    full_prefix = torch.cat([z_ids, latent_end_ids], dim=1)

    answer_ids = model.generate(
        full_prefix,
        max_new_tokens=max_answer_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer_tokens = answer_ids[0, full_prefix.shape[1]:]

    z_list = z_tokens.tolist()
    z_text = tokenizer.decode(z_list, skip_special_tokens=False)
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {
        "question": question,
        "latent_tokens": z_text,
        "answer": answer,
        "z_token_ids": z_list,
        "z_unique": len(set(z_list)),
        "z_total": len(z_list),
    }


def print_result(result, idx=None):
    prefix = f"[{idx}] " if idx is not None else ""
    print(f"\n{'━'*60}")
    print(f"  {prefix}Q: {result['question']}")
    print(f"  A: {result['answer']}")
    print(f"  z ({result['z_unique']}/{result['z_total']} unique):")
    print(f"    {result['latent_tokens'][:200]}")
    # Топ токенов
    top = Counter(result['z_token_ids']).most_common(5)
    from transformers import AutoTokenizer
    print(f"  z top: {top}")
    print(f"{'━'*60}")


def run_inference(checkpoint_dir: str, questions: list = None, latent_len: int = 32):
    print(f"\n{'='*60}")
    print(f"  🧪 ИНФЕРЕНС — {checkpoint_dir}")
    print(f"{'='*60}")

    tokenizer, model = load_student(checkpoint_dir)

    if questions is None:
        questions = [
            "What is the sum of all integers from 1 to 100?",
            "If x² + y² = 25 and x + y = 7, what is xy?",
            "How many ways can you arrange 5 books on a shelf?",
            "What is 17 × 23?",
            "Find the derivative of x³ + 2x² - 5x + 3.",
        ]

    for i, q in enumerate(questions):
        result = generate_answer(tokenizer, model, q, latent_len=latent_len)
        print_result(result, idx=i+1)

    # ── Тест на консистентность: один вопрос 3 раза ──
    print(f"\n{'='*60}")
    print(f"  🔄 ТЕСТ КОНСИСТЕНТНОСТИ (один вопрос, 3 запуска)")
    print(f"{'='*60}")
    test_q = questions[0]
    for run in range(3):
        result = generate_answer(tokenizer, model, test_q, latent_len=latent_len, temperature=0.3)
        z_short = result['latent_tokens'][:100]
        print(f"  run {run+1}: z=[{z_short}...]  →  {result['answer'][:60]}")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/final"
    run_inference(ckpt)
