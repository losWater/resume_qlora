"""
M3: 用 DeepSeek 批量生成「虚构简历文本 + 标准 JSON」配对数据 (蒸馏思路)。
- 全部虚构数据, 无隐私 (合规)。
- 每条让模型先编一份简历纯文本, 再给出对应 schema 的 JSON, 成对保存。
- 通过给每条注入不同的「画像规格」(行业/学历/年限) 来保证多样性, 防过拟合。

运行:  .venv\\Scripts\\python.exe gen_resume_data.py --n 300
"""
import sys, os, json, argparse, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from openai import OpenAI
from schema import validate_label

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
load_dotenv(os.path.join(HERE, ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# 多样性维度: 随机组合, 避免模型只会一种模式
INDUSTRIES = ["互联网/软件", "金融/银行", "电商/零售", "教育培训", "医疗健康",
              "制造业", "市场营销", "人力资源", "土木建筑", "新媒体运营",
              "游戏开发", "人工智能", "财务会计", "法律", "物流供应链"]
DEGREES = ["大专", "本科", "硕士", "博士"]
SENIORITY = ["应届毕业生(0年经验)", "初级(1-3年)", "中级(3-6年)", "高级(6-10年)", "资深/管理(10年以上)"]

SYS_PROMPT = (
    "你是一个用于生成机器学习训练数据的助手。"
    "你需要编造【完全虚构】的中文简历，并给出对应的结构化 JSON 标注。"
    "人名、公司、学校全部虚构，不得使用任何真实个人信息。"
)

# 要求模型一次返回 {resume_text, label_json}
USER_TMPL = """请按以下要求生成一条简历训练样本，行业={industry}，最高学历={degree}，资历={seniority}。

第一步，编写一段**自然、口语化、格式随意**的中文简历纯文本（像真人写的，可包含求职意向、自我评价等噪声信息，长度150-400字）。
第二步，从这段文本中抽取出严格符合下面 schema 的 JSON。

schema:
{{
  "name": 字符串,
  "education": [{{"school":字符串,"degree":字符串,"major":字符串,"year":字符串如"2018-2022"}}],
  "skills": [字符串, ...],
  "work_years": 整数,
  "experience": [{{"company":字符串,"title":字符串,"duration":字符串,"desc":字符串}}]
}}

要求：
- JSON 必须能被 json.loads 解析；work_years 是整数；应届生 work_years=0 且 experience 可为空数组。
- JSON 内容必须与简历文本一致（文本里有的才写进 JSON）。
- 只输出一个 JSON 对象，包含两个键：resume_text(字符串) 和 label_json(对象)。不要任何额外解释或 markdown 代码块。"""


def make_client():
    if not API_KEY:
        print("[错误] 未检测到 DEEPSEEK_API_KEY。请把 Key 填进 .env 后重试。")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def gen_one(client, idx, seed):
    rnd = random.Random(seed + idx)
    industry = rnd.choice(INDUSTRIES)
    degree = rnd.choice(DEGREES)
    seniority = rnd.choice(SENIORITY)
    user = USER_TMPL.format(industry=industry, degree=degree, seniority=seniority)
    for attempt in range(3):  # 简单重试
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYS_PROMPT},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=1.1,  # 高一点增加多样性
                max_tokens=1500,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            text, label = data.get("resume_text"), data.get("label_json")
            if not isinstance(text, str) or not isinstance(label, dict):
                raise ValueError("返回缺 resume_text/label_json")
            if isinstance(label.get("work_years"), float):
                label["work_years"] = int(label["work_years"])
            ok, why = validate_label(label)
            if not ok:
                raise ValueError(f"schema 不合格: {why}")
            return {"resume_text": text.strip(), "label_json": label}
        except Exception as e:
            if attempt == 2:
                return {"_error": f"{type(e).__name__}: {e}"}
            time.sleep(1.5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="生成条数")
    ap.add_argument("--workers", type=int, default=6, help="并发数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    client = make_client()

    print(f"开始生成 {args.n} 条 (model={MODEL}, 并发={args.workers}) ...")
    samples, errors = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, client, i, args.seed): i for i in range(args.n)}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r and "_error" not in r:
                samples.append(r)
            else:
                errors += 1
            if done % 20 == 0 or done == args.n:
                print(f"  进度 {done}/{args.n} | 成功 {len(samples)} | 失败 {errors}")

    # 去重 (按简历文本)
    seen, uniq = set(), []
    for s in samples:
        if s["resume_text"] not in seen:
            seen.add(s["resume_text"]); uniq.append(s)
    random.Random(args.seed).shuffle(uniq)

    n_val = max(1, len(uniq) // 10)  # 9:1
    val, train = uniq[:n_val], uniq[n_val:]

    all_path = os.path.join(DATA_DIR, "resume_all.jsonl")
    train_path = os.path.join(DATA_DIR, "resume_train.jsonl")
    val_path = os.path.join(DATA_DIR, "resume_val.jsonl")
    for path, data in [(all_path, uniq), (train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n完成: 有效 {len(uniq)} 条 (去重后) | 失败 {errors}")
    print(f"  train: {len(train)} -> {train_path}")
    print(f"  val:   {len(val)} -> {val_path}")
    if uniq:
        print("\n样例:")
        print("文本:", uniq[0]["resume_text"][:120], "...")
        print("JSON:", json.dumps(uniq[0]["label_json"], ensure_ascii=False)[:200], "...")


if __name__ == "__main__":
    main()
