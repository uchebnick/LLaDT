import os
import sys
import torch
import torch.multiprocessing as mp

from config import cfg

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
    from actor import run_actor
    from learner import run_learner
    
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
