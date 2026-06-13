"""
Скрипт для проверки гипотезы о том, что латентный язык (z) — это не шум, а осмысленный шифр.
Мы генерируем z, затем случайно искажаем (пертурбируем) часть токенов в нём,
и смотрим, как это ломает финальный ответ Студента.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import random

def load_student(checkpoint_dir: str, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, f"{checkpoint_dir}/student")
    model.eval()
    return tokenizer, model

@torch.no_grad()
def answer_with_perturbation(tokenizer, model, question: str, perturb_prob=0.3, latent_len=128, max_ans=64) -> dict:
    device = next(model.parameters()).device

    # Шаг 1: префикс до z
    prefix_text = f"Q: {question}\n<think>\n"
    prefix_ids  = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)

    # Шаг 2: Студент генерирует z (как априорное распределение p(z|Q))
    z_out = model.generate(
        prefix_ids,
        max_new_tokens=latent_len,
        min_new_tokens=latent_len,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None,
    )
    z_ids = z_out[0, prefix_ids.shape[1]:]

    # Шаг 3: Получаем оригинальный ответ (без искажений)
    suffix_ids = tokenizer("\n</think>\nA: ", add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    
    full_clean = torch.cat([prefix_ids, z_ids.unsqueeze(0), suffix_ids], dim=1)
    ans_out_clean = model.generate(
        full_clean,
        max_new_tokens=max_ans,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    ans_clean_ids = ans_out_clean[0, full_clean.shape[1]:]

    # Шаг 4: ПЕРТУРБАЦИЯ (искажаем часть токенов z)
    # Наш алфавит — в основном цифры. Возьмем токены цифр от '0' до '9'.
    digit_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    
    z_ids_perturbed = z_ids.clone()
    num_to_perturb = int(len(z_ids) * perturb_prob)
    indices_to_perturb = random.sample(range(len(z_ids)), num_to_perturb)
    
    for idx in indices_to_perturb:
        z_ids_perturbed[idx] = random.choice(digit_tokens)

    # Шаг 5: Получаем ответ с искаженным z
    full_perturbed = torch.cat([prefix_ids, z_ids_perturbed.unsqueeze(0), suffix_ids], dim=1)
    ans_out_perturbed = model.generate(
        full_perturbed,
        max_new_tokens=max_ans,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    ans_perturbed_ids = ans_out_perturbed[0, full_perturbed.shape[1]:]

    return {
        "question": question,
        "z_clean": tokenizer.decode(z_ids, skip_special_tokens=False),
        "z_perturbed": tokenizer.decode(z_ids_perturbed, skip_special_tokens=False),
        "ans_clean": tokenizer.decode(ans_clean_ids, skip_special_tokens=True).strip(),
        "ans_perturbed": tokenizer.decode(ans_perturbed_ids, skip_special_tokens=True).strip(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./kaggle_logs/checkpoints/step_1000", help="Path to checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prob", type=float, default=0.3, help="Probability of corrupting a token in z")
    args = parser.parse_args()

    tokenizer, model = load_student(args.checkpoint, args.device)

    tests = [
        "What is the sum of all integers from 1 to 100?",
        "If x^2 = 64 and x > 0, find x.",
        "Solve for y: 3y - 7 = 14.",
    ]
    
    print(f"=== PERTURBATION TEST (Corrupting {args.prob*100}% of latent tokens) ===")
    for q in tests:
        r = answer_with_perturbation(tokenizer, model, q, perturb_prob=args.prob)
        print(f"\nQ: {r['question']}")
        print(f"Clean Answer: {r['ans_clean']}")
        print(f"Perturbed Answer: {r['ans_perturbed']}")
        print(f"---")
        print(f"z_clean: {r['z_clean'][:60]}...")
        print(f"z_perturbed: {r['z_perturbed'][:60]}...")
