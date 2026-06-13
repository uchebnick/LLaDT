import time
import torch
from config import cfg
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
            for item in batch:
                text = f"Q: {item['q'][:400]}\nA: {item['a'][:80]}\n<think>\n"
                ids = tokenizer(text, add_special_tokens=True)["input_ids"]
                prompts.append(ids[:cfg.max_q_tokens + cfg.max_a_tokens + 10])
                
            mp_len = max(len(p) for p in prompts)
            B = len(batch)
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
            for i in range(B):
                item = {"q": batch["q"][i] if isinstance(batch, dict) else batch[i]["q"], 
                        "a": batch["a"][i] if isinstance(batch, dict) else batch[i]["a"]}
                # Если очередь переполнена, put() заблокирует выполнение до освобождения места
                data_queue.put((item, z_list[i]))
                
            t1 = time.time()
            if step % 5 == 0:
                print(f"[Actor] Generated {B} samples in {t1-t0:.2f}s | Queue size: ~{data_queue.qsize()}", flush=True)
            step += 1
