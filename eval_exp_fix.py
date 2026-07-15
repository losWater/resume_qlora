"""
P1 经历专项 第三步: 分离"评估修正"与"真实提升"。

三组对比 (同一套指标):
  ① 旧 adapter × 旧 val gold  —— 升级前口径 (直接读 outputs/ood_eval_result.json, 不重跑)
  ② 旧 adapter × 新 val gold  —— ①→② 的差 = 评估修正 (标注噪声压低了多少分)
  ③ 新 adapter × 新 val gold  —— ②→③ 的差 = 真实提升 (修复训练数据带来的)
另附: 新 adapter × OOD, 与旧 adapter 的 74.4% 对比泛化变化。

运行:  .venv\\Scripts\\python.exe eval_exp_fix.py
产物:  outputs/exp_fix_result.json
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
OLD_ADAPTER = os.path.join(HERE, "outputs", "resume_adapter")
NEW_ADAPTER = os.path.join(HERE, "outputs", "resume_adapter_v2")
VAL_FILE = os.path.join(HERE, "data", "resume_val.jsonl")      # 已是重标后的新 gold
OOD_FILE = os.path.join(HERE, "data", "resume_ood.jsonl")
OUT_FILE = os.path.join(HERE, "outputs", "exp_fix_result.json")

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
    old_result = json.load(open(os.path.join(HERE, "outputs", "ood_eval_result.json"),
                                encoding="utf-8"))
    stage1 = old_result["in_dist"]          # ① 旧 adapter × 旧 gold
    old_ood = old_result["ood"]

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

    # ② 旧 adapter × 新 gold
    model = PeftModel.from_pretrained(base, OLD_ADAPTER)
    model.eval()
    stage2 = evaluate(model, tok, val_rows, "② 旧 adapter × 新 gold")

    # ③ 新 adapter × 新 gold (+ OOD)
    del model
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(base, NEW_ADAPTER)
    model.eval()
    stage3 = evaluate(model, tok, val_rows, "③ 新 adapter × 新 gold")
    new_ood = evaluate(model, tok, ood_rows, "③ 新 adapter × OOD")

    print("\n" + "=" * 78)
    print("效应分离: ① 旧模型×旧gold | ② 旧模型×新gold | ③ 新模型×新gold")
    print("  ①→② = 评估修正 (标注噪声)   ②→③ = 真实提升 (修复训练数据)")
    print("=" * 78)
    print(f"{'指标':<16}{'①':>9}{'②':>9}{'③':>9}{'评估修正':>10}{'真实提升':>10}")
    print("-" * 66)
    table = []
    for name, key in LABELS:
        a, b, c = stage1[key], stage2[key], stage3[key]
        print(f"{name:<16}{a*100:>8.1f}%{b*100:>8.1f}%{c*100:>8.1f}%"
              f"{(b-a)*100:>+9.1f}pp{(c-b)*100:>+9.1f}pp")
        table.append({"metric": name, "s1": round(a, 4), "s2": round(b, 4),
                      "s3": round(c, 4), "eval_correction_pp": round((b-a)*100, 1),
                      "true_gain_pp": round((c-b)*100, 1)})

    print(f"\nOOD 平均 F1: 旧 adapter {old_ood['avg_list_f1']*100:.1f}% "
          f"-> 新 adapter {new_ood['avg_list_f1']*100:.1f}%")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "note": "①旧adapter×旧gold(引自ood_eval_result.json) ②旧adapter×新gold ③新adapter×新gold",
            "stage1_old_model_old_gold": stage1,
            "stage2_old_model_new_gold": stage2,
            "stage3_new_model_new_gold": stage3,
            "ood_old_adapter": old_ood,
            "ood_new_adapter": new_ood,
            "table": table,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {OUT_FILE}")


if __name__ == "__main__":
    main()
