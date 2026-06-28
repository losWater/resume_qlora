"""
M2: 用 QLoRA 微调 Qwen2.5-1.5B-Instruct 做中文 NER (文本 -> JSON 实体)。
目的: 跑通"文本->JSON"的微调范式, 为简历抽取 (M4) 打基础。

链路: 4-bit 量化加载基座 + LoRA adapter + trl SFTTrainer, 只对答案算 loss。

运行:  .venv\\Scripts\\python.exe train_ner_qlora.py
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

MODEL_PATH = r"D:\cv_view\resume_qlora\models\Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = r"D:\cv_view\resume_qlora\outputs\ner_adapter"

INSTRUCTION = (
    "你是命名实体识别助手。请从下面的句子中抽取所有实体，"
    "以 JSON 数组输出，每个元素形如 {\"entity\": \"实体文本\", \"type\": \"类型\"}，"
    "type 只能是 人名/地名/机构名 之一。只输出 JSON，不要多余解释。\n句子："
)


def main():
    # ---------- 1. tokenizer ----------
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- 2. 4-bit 量化加载基座 (QLoRA 的 "Q") ----------
    # nf4 量化 + bf16 计算 + 双重量化, 8GB 显存关键
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.config.use_cache = False  # 训练 + 梯度检查点时必须关

    # ---------- 3. LoRA 配置 (QLoRA 的 "LoRA") ----------
    # 只训练注意力的 q/k/v/o 投影层的低秩增量, 原权重冻结
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none",
    )

    # ---------- 4. 数据: 转成对话式 prompt/completion ----------
    # completion_only_loss=True 会只对 completion(答案 JSON) 算 loss, 屏蔽问题
    def to_chat(ex):
        user = INSTRUCTION + ex["text"]
        answer = json.dumps(ex["entities"], ensure_ascii=False)
        return {
            "prompt": [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": answer}],
        }

    ds = load_dataset("json", data_files={
        "train": r"D:\cv_view\resume_qlora\data\ner_train.jsonl",
        "validation": r"D:\cv_view\resume_qlora\data\ner_val.jsonl",
    })
    cols = ds["train"].column_names
    ds = ds.map(to_chat, remove_columns=cols)

    # ---------- 5. 训练配置 (8GB 适配) ----------
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=512,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=1,      # 8GB 显存: batch 压到 1, 防爆显存
        per_device_eval_batch_size=1,       # 验证默认是 8, 必须也压到 1, 否则 eval 时易 OOM
        gradient_accumulation_steps=8,      # 有效 batch = 8 (1*8), 用累积换显存
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",                   # M2 先不接 swanlab, M4 再接
        optim="paged_adamw_8bit",           # 配合 4-bit, 省显存
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        peft_config=lora,
    )

    print("=" * 60)
    print("开始训练 (M2 NER QLoRA)")
    print("=" * 60)
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\n训练完成，adapter 已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
