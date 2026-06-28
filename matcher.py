"""
M6: 岗位匹配模块 (⚙️ 规则, 不训练)。
- 简历 -> JSON: 用微调后的模型 (adapter)
- 岗位 JD -> JSON: 用原始基座 (disable_adapter), JD 解析较简单, 零样本够用
- 规则打分: 技能重合率 + 年限达标 + 学历达标, 加权求和, 给逐项理由 (可解释)

运行 (内置 demo):  .venv\\Scripts\\python.exe matcher.py
"""
import sys, json, re
from contextlib import nullcontext
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from schema import INSTRUCTION

MODEL_PATH = r"D:\cv_view\resume_qlora\models\Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = r"D:\cv_view\resume_qlora\outputs\resume_adapter"

# 学历等级 (用于"达标"比较)
DEGREE_RANK = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}

# 打分权重 (可配置)
WEIGHTS = {"skills": 0.6, "years": 0.2, "degree": 0.2}

JD_INSTRUCTION = (
    "你是招聘 JD 解析助手。请从下面的岗位描述中抽取要求，输出 JSON，字段：\n"
    "- required_skills: 要求的技能(字符串数组)\n"
    "- min_work_years: 要求的最低工作年限(整数，无明确要求填0)\n"
    "- min_degree: 要求的最低学历(只能是 大专/本科/硕士/博士 之一，无要求填\"大专\")\n"
    "只输出 JSON，不要多余解释。\nJD："
)


# ---------------- 工具 ----------------
def norm(s):
    return str(s).strip().lower()


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def degree_rank_of_resume(resume):
    """取简历里最高学历的等级。"""
    best = 0
    for e in resume.get("education", []):
        best = max(best, DEGREE_RANK.get(str(e.get("degree", "")).strip(), 0))
    return best


# ---------------- 模型解析 ----------------
class ResumeJDParser:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda",
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        self.model.eval()

    def _gen(self, instruction, text, use_adapter):
        msgs = [{"role": "user", "content": instruction + text}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        # use_adapter=False 时临时关掉 LoRA, 用纯基座解析 JD
        ctx = nullcontext() if use_adapter else self.model.disable_adapter()
        with torch.no_grad(), ctx:
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    def parse_resume(self, resume_text):
        return parse_json(self._gen(INSTRUCTION, resume_text, use_adapter=True))

    def parse_jd(self, jd_text):
        return parse_json(self._gen(JD_INSTRUCTION, jd_text, use_adapter=False))


# ---------------- 规则打分 ----------------
def match_score(resume, jd):
    reasons = []

    # 1. 技能匹配度: JD 要求技能里有多少在简历中命中 (模糊: 互相包含即算命中)
    jd_skills = [s for s in jd.get("required_skills", []) if isinstance(s, str)]
    res_skills = [s for s in resume.get("skills", []) if isinstance(s, str)]
    hit, miss = [], []
    for js in jd_skills:
        matched = any(norm(js) in norm(rs) or norm(rs) in norm(js) for rs in res_skills)
        (hit if matched else miss).append(js)
    skill_score = len(hit) / len(jd_skills) if jd_skills else 1.0
    reasons.append(f"技能: 命中 {len(hit)}/{len(jd_skills)}" +
                   (f"（命中: {', '.join(hit)}）" if hit else "") +
                   (f"（缺失: {', '.join(miss)}）" if miss else ""))

    # 2. 年限达标
    res_years = resume.get("work_years", 0)
    try:
        res_years = int(res_years)
    except Exception:
        res_years = 0
    min_years = jd.get("min_work_years", 0)
    try:
        min_years = int(min_years)
    except Exception:
        min_years = 0
    years_ok = res_years >= min_years
    years_score = 1.0 if years_ok else (res_years / min_years if min_years else 1.0)
    reasons.append(f"年限: 要求 {min_years} 年, 简历 {res_years} 年 -> " +
                   ("达标" if years_ok else "不足"))

    # 3. 学历达标
    res_rank = degree_rank_of_resume(resume)
    min_rank = DEGREE_RANK.get(str(jd.get("min_degree", "大专")).strip(), 1)
    degree_ok = res_rank >= min_rank
    degree_score = 1.0 if degree_ok else 0.0
    inv = {v: k for k, v in DEGREE_RANK.items()}
    reasons.append(f"学历: 要求 {inv.get(min_rank,'?')}, 简历最高 {inv.get(res_rank,'无')} -> " +
                   ("达标" if degree_ok else "不达标"))

    total = (WEIGHTS["skills"] * skill_score +
             WEIGHTS["years"] * years_score +
             WEIGHTS["degree"] * degree_score) * 100
    return {
        "score": round(total, 1),
        "sub_scores": {
            "skills": round(skill_score * 100, 1),
            "years": round(years_score * 100, 1),
            "degree": round(degree_score * 100, 1),
        },
        "reasons": reasons,
    }


# ---------------- demo ----------------
DEMO_RESUME = (
    "我叫王强，硕士毕业于江城大学计算机专业（2017-2020），工作5年。"
    "目前在一家互联网公司做后端开发，熟悉 Python、Java、MySQL、Redis 和分布式系统设计，"
    "带过3人小团队。之前在两家创业公司做过开发。求职意向：高级后端工程师。"
)
DEMO_JD = (
    "招聘高级后端工程师。要求：本科及以上学历，3年以上后端开发经验，"
    "精通 Python 和 MySQL，熟悉 Redis、消息队列，有分布式系统经验者优先。"
)


def main():
    print("加载模型 (基座 + adapter) ...")
    parser = ResumeJDParser()

    print("\n=== 解析简历 (微调模型) ===")
    resume = parser.parse_resume(DEMO_RESUME)
    print(json.dumps(resume, ensure_ascii=False, indent=2))

    print("\n=== 解析 JD (原始基座) ===")
    jd = parser.parse_jd(DEMO_JD)
    print(json.dumps(jd, ensure_ascii=False, indent=2))

    print("\n=== 匹配结果 ===")
    result = match_score(resume, jd)
    print(f"匹配分: {result['score']} / 100")
    print(f"分项: 技能 {result['sub_scores']['skills']} | "
          f"年限 {result['sub_scores']['years']} | 学历 {result['sub_scores']['degree']}")
    print("理由:")
    for r in result["reasons"]:
        print("  -", r)


if __name__ == "__main__":
    main()
