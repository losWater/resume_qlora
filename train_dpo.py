"""
M10 第二步: DPO 训练。
标准流程 SFT -> DPO: 在 4-bit 基座 + 已训好的 SFT adapter 之上继续做偏好对齐,
让模型不只知道"什么是对的"(SFT), 还知道"什么是错的"(偏好对里的 rejected)。

8GB 显存关键点: DPO 需要 policy 和 reference 两份 logits。trl 对 PEFT 模型
支持 ref_model=None —— 计算 reference logits 时临时 disable_adapter(),
一份模型双角色, 不用加载两份权重。

运行:  .venv\\Scripts\\python.exe train_dpo.py
产物:  outputs/dpo_adapter/ (在 SFT adapter 基础上继续训练得到, 入库)
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "Qwen2.5-1.5B-Instruct")
SFT_ADAPTER = os.path.join(HERE, "outputs", "resume_adapter")
DPO_FILE = os.path.join(HERE, "data", "resume_dpo.jsonl")
OUTPUT_DIR = os.path.join(HERE, "outputs", "dpo_adapter")
SEED = 42


def main():
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda",
    )
    base.config.use_cache = False

    # 在 SFT adapter 之上继续训练 (is_trainable=True), 而不是新开一个随机 adapter
    model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)

    # 偏好对 -> 对话式 prompt/chosen/rejected
    def to_chat(ex):
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "chosen": [{"role": "assistant", "content": ex["chosen"]}],
            "rejected": [{"role": "assistant", "content": ex["rejected"]}],
        }

    ds = load_dataset("json", data_files=DPO_FILE)["train"]
    ds = ds.map(to_chat, remove_columns=ds.column_names)
    print(f"偏好对: {len(ds)} 条")

    cfg = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,                        # 偏离 reference 的惩罚强度, 经验起点
        learning_rate=1e-5,              # DPO 比 SFT 低一个数量级
        num_train_epochs=2,
        max_length=800,                  # 实测最长样本 772 token, 800 零截断且比 1024 省显存
        precompute_ref_log_probs=True,   # ref logps 训练前一次算完, 训练循环省一半前向/显存
                                         # (8GB 卡上不开会顶爆显存溢出到共享内存, 慢两个数量级)
        per_device_train_batch_size=1,   # DPO 一步要算 chosen+rejected 两份, 显存吃紧
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
        seed=SEED,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                  # PEFT 模型: disable_adapter 当 reference
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    print("=" * 60)
    print("开始 DPO 训练 (SFT adapter 之上继续对齐)")
    print("=" * 60)
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\nDPO 训练完成, adapter 已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
