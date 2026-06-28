"""
M4: 用 QLoRA 微调 Qwen2.5-1.5B-Instruct 做简历结构化抽取 (简历文本 -> JSON)。
复用 M2 的链路, 只换数据/指令/输出目录, max_length 调大 (简历比 NER 句子长)。

运行:  .venv\\Scripts\\python.exe train_resume_qlora.py
"""
import sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from schema import INSTRUCTION

MODEL_PATH = r"D:\cv_view\resume_qlora\models\Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = r"D:\cv_view\resume_qlora\outputs\resume_adapter"
TRAIN_FILE = r"D:\cv_view\resume_qlora\data\resume_train.jsonl"
VAL_FILE = r"D:\cv_view\resume_qlora\data\resume_val.jsonl"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit 量化加载基座 (QLoRA 的 Q)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda",
    )
    model.config.use_cache = False

    # LoRA (QLoRA 的 LoRA): 只训注意力 q/k/v/o
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none",
    )

    # 数据 -> 对话式 prompt/completion, 只对答案 JSON 算 loss
    def to_chat(ex):
        user = INSTRUCTION + ex["resume_text"]
        answer = json.dumps(ex["label_json"], ensure_ascii=False)
        return {
            "prompt": [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": answer}],
        }

    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    ds = ds.map(to_chat, remove_columns=ds["train"].column_names)

    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=1024,                    # 简历比 NER 长, 调大
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=1,      # 8GB 防爆显存
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,      # 有效 batch = 8
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        processing_class=tok, peft_config=lora,
    )

    print("=" * 60)
    print("开始训练 (M4 简历抽取 QLoRA)")
    print("=" * 60)
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\n训练完成，adapter 已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
