"""
M2: 加载基座 + 训练好的 LoRA adapter, 对新句子做 NER 推理, 输出 JSON。
对比验证: 微调是否真的让模型学会了"文本->JSON实体"的格式与抽取。

运行:  .venv\\Scripts\\python.exe infer_ner.py
"""
import sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = r"D:\cv_view\resume_qlora\models\Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = r"D:\cv_view\resume_qlora\outputs\ner_adapter"

INSTRUCTION = (
    "你是命名实体识别助手。请从下面的句子中抽取所有实体，"
    "以 JSON 数组输出，每个元素形如 {\"entity\": \"实体文本\", \"type\": \"类型\"}，"
    "type 只能是 人名/地名/机构名 之一。只输出 JSON，不要多余解释。\n句子："
)

TEST_SENTENCES = [
    "马云创办的阿里巴巴总部在杭州。",                      # 训练里没出现过的人名/搭配
    "去年我从西安搬到了厦门，进了华为公司。",
    "李娜和周婷一起去南京大学参观。",
]


def build_prompt(tok, sentence):
    msgs = [{"role": "user", "content": INSTRUCTION + sentence}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tok, sentence):
    text = build_prompt(tok, sentence)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main():
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
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    print("=" * 60)
    print("微调后模型 NER 推理结果")
    print("=" * 60)
    ok = 0
    for s in TEST_SENTENCES:
        raw = generate(model, tok, s)
        print(f"\n句子: {s}")
        print(f"输出: {raw}")
        try:
            parsed = json.loads(raw)
            print(f"[JSON合法] 解析出 {len(parsed)} 个实体")
            ok += 1
        except Exception as e:
            print(f"[JSON非法] {e}")
    print(f"\n合法率: {ok}/{len(TEST_SENTENCES)}")


if __name__ == "__main__":
    main()
