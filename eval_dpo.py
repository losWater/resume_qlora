"""
M10 第三步: DPO 后模型 vs SFT-only 对比评估。
同一套指标 (口径与 evaluate.py 一致), 在同源 val (29 条) 和 OOD (40 条) 都跑。
重点看: 工作经历字段 (脑补问题) 和 JSON 可解析率有没有改善, 其他字段有没有退步。

运行:  .venv\\Scripts\\python.exe eval_dpo.py
产物:  outputs/dpo_eval_result.json
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from evaluate import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "Qwen2.5-1.5B-Instruct")
DPO_ADAPTER = os.path.join(HERE, "outputs", "dpo_adapter")
VAL_FILE = os.path.join(HERE, "data", "resume_val.jsonl")
OOD_FILE = os.path.join(HERE, "data", "resume_ood.jsonl")
SFT_RESULT = os.path.join(HERE, "outputs", "ood_eval_result.json")
OUT_FILE = os.path.join(HERE, "outputs", "dpo_eval_result.json")

LABELS = [
    ("JSON 可解析率", "json_parse_rate"),
    ("姓名准确率", "name_acc"),
    ("工作年限准确率", "work_years_acc"),
    ("技能 F1", "skills_f1"),
    ("教育经历 F1", "education_f1"),
    ("工作经历 F1", "experience_f1"),
    ("列表字段平均 F1", "avg_list_f1"),
]


def main():
    val_rows = [json.loads(l) for l in open(VAL_FILE, encoding="utf-8")]
    ood_rows = [json.loads(l) for l in open(OOD_FILE, encoding="utf-8")]
    sft = json.load(open(SFT_RESULT, encoding="utf-8"))  # SFT-only 的两集结果

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
    model = PeftModel.from_pretrained(base, DPO_ADAPTER)
    model.eval()

    dpo_val = evaluate(model, tok, val_rows, f"DPO 同源 val ({len(val_rows)} 条)")
    dpo_ood = evaluate(model, tok, ood_rows, f"DPO OOD ({len(ood_rows)} 条)")

    def show(title, sft_res, dpo_res):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print(f"{'指标':<16}{'SFT':>10}{'SFT+DPO':>10}{'差值':>12}")
        print("-" * 50)
        table = []
        for name, key in LABELS:
            a, b = sft_res[key], dpo_res[key]
            print(f"{name:<16}{a*100:>9.1f}%{b*100:>9.1f}%{(b-a)*100:>+10.1f}pp")
            table.append({"metric": name, "sft": round(a, 4), "dpo": round(b, 4),
                          "delta_pp": round((b - a) * 100, 1)})
        return table

    tbl_val = show("同源 val: SFT vs SFT+DPO", sft["in_dist"], dpo_val)
    tbl_ood = show("OOD: SFT vs SFT+DPO", sft["ood"], dpo_ood)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "dpo_adapter": "outputs/dpo_adapter (SFT adapter 之上继续 DPO)",
            "dpo_config": {"beta": 0.1, "lr": 1e-5, "epochs": 2,
                           "pairs_file": "data/resume_dpo.jsonl"},
            "in_dist": {"sft": sft["in_dist"], "dpo": dpo_val, "table": tbl_val},
            "ood": {"sft": sft["ood"], "dpo": dpo_ood, "table": tbl_ood},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {OUT_FILE}")


if __name__ == "__main__":
    main()
