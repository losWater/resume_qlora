"""
M8: 超参数消融实验。
以当前配置 (r=8, alpha=16, QKVO, lr=2e-4) 为中心, 控制变量法每次只动一个维度:
  - r ∈ {4, 8, 16}  (alpha 跟随 = 2r)
  - target_modules ∈ {QKVO, QKVO+MLP(gate/up/down_proj)}
  - lr ∈ {1e-4, 2e-4, 5e-4}
去重后共 6 组配置。每组 = 训练 + 评估(微调后, 同源 val 29 条) + 记录。

设计说明:
- 每组训练放在独立子进程里跑 (本脚本以 --worker N 方式自调用),
  保证显存彻底释放, 且单组失败/中断后重跑会自动跳过已完成的组。
- 固定随机种子 42, 所有组共用同一份 train/val。
- 评估逻辑直接 import evaluate.py, 与 M5 基线口径完全一致。

运行:  .venv\\Scripts\\python.exe run_ablation.py
产物:  outputs/ablation_results.json (汇总) + outputs/ablation/<组名>/ (各组 adapter, 不入库)
"""
import sys, os, json, time, subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "Qwen2.5-1.5B-Instruct")
TRAIN_FILE = os.path.join(BASE, "data", "resume_train.jsonl")
VAL_FILE = os.path.join(BASE, "data", "resume_val.jsonl")
ABLATION_DIR = os.path.join(BASE, "outputs", "ablation")
RESULT_FILE = os.path.join(BASE, "outputs", "ablation_results.json")

SEED = 42
QKVO = ["q_proj", "k_proj", "v_proj", "o_proj"]
QKVO_MLP = QKVO + ["gate_proj", "up_proj", "down_proj"]

# 控制变量: 中心配置 = baseline, 其余每组只改一个维度
CONFIGS = [
    {"name": "baseline_r8",  "r": 8,  "alpha": 16, "targets": QKVO,     "lr": 2e-4},
    {"name": "r4",           "r": 4,  "alpha": 8,  "targets": QKVO,     "lr": 2e-4},
    {"name": "r16",          "r": 16, "alpha": 32, "targets": QKVO,     "lr": 2e-4},
    {"name": "qkvo_mlp",     "r": 8,  "alpha": 16, "targets": QKVO_MLP, "lr": 2e-4},
    {"name": "lr1e-4",       "r": 8,  "alpha": 16, "targets": QKVO,     "lr": 1e-4},
    {"name": "lr5e-4",       "r": 8,  "alpha": 16, "targets": QKVO,     "lr": 5e-4},
]


def result_path(name):
    return os.path.join(ABLATION_DIR, f"{name}_result.json")


# ---------------- worker: 单组 训练 + 评估 ----------------
def run_one(cfg):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, PeftModel
    from trl import SFTTrainer, SFTConfig
    from schema import INSTRUCTION
    from evaluate import evaluate as eval_metrics

    set_seed(SEED)
    out_dir = os.path.join(ABLATION_DIR, cfg["name"])

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05,
        target_modules=cfg["targets"], task_type="CAUSAL_LM", bias="none",
    )

    def to_chat(ex):
        user = INSTRUCTION + ex["resume_text"]
        answer = json.dumps(ex["label_json"], ensure_ascii=False)
        return {
            "prompt": [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": answer}],
        }

    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    ds = ds.map(to_chat, remove_columns=ds["train"].column_names)

    sft_cfg = SFTConfig(
        output_dir=out_dir,
        max_length=1024,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=cfg["lr"],
        num_train_epochs=3,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="no",          # 消融只看最终指标, 省掉逐 epoch eval
        save_strategy="no",          # 只存最终 adapter
        report_to="none",
        optim="paged_adamw_8bit",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model, args=sft_cfg,
        train_dataset=ds["train"], eval_dataset=None,
        processing_class=tok, peft_config=lora,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    trainer.save_model(out_dir)

    # 释放训练显存, 重新以 [4bit 基座 + adapter] 干净加载后评估, 与 M5 口径一致
    del trainer, model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda",
    )
    ft = PeftModel.from_pretrained(base, out_dir)
    ft.eval()
    rows = [json.loads(l) for l in open(VAL_FILE, encoding="utf-8")]
    metrics = eval_metrics(ft, tok, rows, cfg["name"])

    result = {
        "config": {
            "name": cfg["name"], "r": cfg["r"], "lora_alpha": cfg["alpha"],
            "target_modules": cfg["targets"], "learning_rate": cfg["lr"],
            "epochs": 3, "seed": SEED,
        },
        "trainable_params": trainable,
        "train_seconds": round(train_seconds, 1),
        "peak_vram_gb": round(peak_mem_gb, 2),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    with open(result_path(cfg["name"]), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[{cfg['name']}] 完成: 训练 {train_seconds/60:.1f} 分钟, "
          f"峰值显存 {peak_mem_gb:.2f} GB, avg_list_f1 = {metrics['avg_list_f1']:.4f}")


# ---------------- 主进程: 调度 + 汇总 ----------------
def aggregate():
    results = []
    for cfg in CONFIGS:
        p = result_path(cfg["name"])
        if os.path.exists(p):
            results.append(json.load(open(p, encoding="utf-8")))
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "val_file": "data/resume_val.jsonl",
                   "note": "控制变量消融: 每组只改一个维度; 评估口径与 outputs/eval_result.json 一致",
                   "results": results}, f, ensure_ascii=False, indent=2)

    # markdown 对比表
    print("\n" + "=" * 100)
    print("消融汇总 (已完成 %d/%d 组)" % (len(results), len(CONFIGS)))
    print("=" * 100)
    header = ("| 配置 | r | alpha | 挂载层 | lr | 可训练参数 | 训练时长 | 峰值显存 "
              "| 可解析率 | 技能F1 | 教育F1 | 经历F1 | 平均F1 |")
    sep = "|" + "---|" * 13
    lines = [header, sep]
    for r in results:
        c, m = r["config"], r["metrics"]
        tm = "QKVO+MLP" if len(c["target_modules"]) > 4 else "QKVO"
        lines.append(
            f"| {c['name']} | {c['r']} | {c['lora_alpha']} | {tm} | {c['learning_rate']:g} "
            f"| {r['trainable_params']/1e6:.1f}M | {r['train_seconds']/60:.1f}min | {r['peak_vram_gb']}GB "
            f"| {m['json_parse_rate']*100:.1f}% | {m['skills_f1']*100:.1f}% | {m['education_f1']*100:.1f}% "
            f"| {m['experience_f1']*100:.1f}% | {m['avg_list_f1']*100:.1f}% |")
    md = "\n".join(lines)
    print(md)
    print(f"\n汇总已保存: {RESULT_FILE}")
    return md


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        run_one(CONFIGS[int(sys.argv[2])])
        return

    os.makedirs(ABLATION_DIR, exist_ok=True)
    for i, cfg in enumerate(CONFIGS):
        if os.path.exists(result_path(cfg["name"])):
            print(f"[{i+1}/{len(CONFIGS)}] {cfg['name']} 已有结果, 跳过")
            continue
        print(f"\n[{i+1}/{len(CONFIGS)}] 开始: {cfg['name']}")
        ret = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", str(i)])
        if ret.returncode != 0:
            print(f"[{cfg['name']}] 失败 (exit {ret.returncode}), 中止。修复后重跑本脚本会跳过已完成的组。")
            sys.exit(1)
    aggregate()


if __name__ == "__main__":
    main()
