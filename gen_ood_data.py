"""
M9: 生成分布外 (OOD) 验证集, 量化"同源评估偏乐观"的程度。

与训练数据 (gen_resume_data.py) 相比, 两条去同源手段同时用上:
1. 换生成模型: deepseek-chat -> deepseek-reasoner (可用 OOD_MODEL 环境变量覆盖);
2. 换 prompt 模板: 要求生成更接近真实简历的"脏"文本——口语化流水账、字段缺失、
   时间线断档、公司缩写、技能夹在经历描述里、错别字、时间格式混乱等。

标注仍严格遵守 schema: 文本里没有的信息不编造, 缺失字段用空串/空数组,
生成后过 validate_label + 学历规范化, 再人工抽查确认。

运行:  .venv\\Scripts\\python.exe gen_ood_data.py --n 40
产物:  data/resume_ood.jsonl (全合成无隐私, 入库)
"""
import sys, os, json, argparse, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from openai import OpenAI
from schema import validate_label
from clean_resume_data import normalize_degree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
load_dotenv(os.path.join(HERE, ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("OOD_MODEL", "deepseek-reasoner").strip()  # 与训练数据的 deepseek-chat 不同

# 与训练集相同的画像维度 (保持任务相同), 但叠加"脏格式"特征 (分布不同)
INDUSTRIES = ["互联网/软件", "金融/银行", "电商/零售", "教育培训", "医疗健康",
              "制造业", "市场营销", "人力资源", "土木建筑", "新媒体运营",
              "游戏开发", "人工智能", "财务会计", "法律", "物流供应链"]
DEGREES = ["大专", "本科", "硕士", "博士"]
SENIORITY = ["应届毕业生(0年经验)", "初级(1-3年)", "中级(3-6年)", "高级(6-10年)", "资深/管理(10年以上)"]

# 每条样本随机注入 2~3 个"脏"特征, 模拟真实简历的不规整
DIRTY_TRAITS = [
    "整体是口语化流水账自我介绍，没有小标题、不分段，像微信里发的一段话",
    "教育经历信息不全：没写就读年份，或没写专业（JSON 对应字段留空字符串）",
    "工作时间线有断档（中间有一两年空白，文本可顺带解释如'回老家了一段时间'）",
    "公司名用缩写或俗称（虚构的，比如'在鹏达干了三年'而不写全称'鹏达科技有限公司'，JSON 里 company 写文本中出现的叫法）",
    "技能不单独罗列，而是散落在工作经历的描述句子里",
    "包含 2~3 个常见错别字（如'工做''在只期间''负则'），JSON 标注仍写正确内容",
    "夹杂与抽取无关的内容：期望薪资、求职意向、'吃苦耐劳'式自我评价",
    "时间格式混乱不统一（'2019.3'、'19年6月'、'2020-2021'混着用，duration/year 照抄文本写法）",
]

SYS_PROMPT = (
    "你是一个用于生成机器学习评估数据的助手。"
    "你需要编造【完全虚构】的中文简历，并给出对应的结构化 JSON 标注。"
    "人名、公司、学校全部虚构，不得使用任何真实个人信息。"
)

USER_TMPL = """请生成一条用于评估简历抽取模型的样本，行业={industry}，最高学历={degree}，资历={seniority}。

第一步，编写一段中文简历纯文本（200-450字）。这段文本要模拟【真实但不规整】的简历，必须体现以下特征：
{traits}

第二步，从这段文本中抽取出严格符合下面 schema 的 JSON 标注（这是 ground truth，必须准确）。

schema:
{{
  "name": 字符串,
  "education": [{{"school":字符串,"degree":字符串,"major":字符串,"year":字符串}}],
  "skills": [字符串, ...],
  "work_years": 整数,
  "experience": [{{"company":字符串,"title":字符串,"duration":字符串,"desc":字符串}}]
}}

标注要求（严格执行）：
- JSON 必须能被 json.loads 解析；work_years 是整数（文本没明说就按经历时间合理推算）。
- 文本里没有的信息绝不编造：缺的字段填空字符串 ""，应届生 experience 可为空数组。
- 技能散落在描述里的，也要抽进 skills 数组。
- 公司/学校名照抄文本中的写法（文本用缩写标注就写缩写）。
- 错别字只出现在简历文本里，JSON 标注写正确的词。
- 只输出一个 JSON 对象，包含两个键：resume_text(字符串) 和 label_json(对象)。不要任何额外解释或 markdown 代码块。"""


def parse_json_loose(raw: str):
    """reasoner 可能带 markdown 围栏或前后缀, 宽松解析。"""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def make_client():
    if not API_KEY:
        print("[错误] 未检测到 DEEPSEEK_API_KEY。请把 Key 填进 .env 后重试。")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def gen_one(client, idx, seed):
    rnd = random.Random(seed * 1000 + idx)   # 与训练集的 seed+idx 错开
    industry = rnd.choice(INDUSTRIES)
    degree = rnd.choice(DEGREES)
    seniority = rnd.choice(SENIORITY)
    traits = rnd.sample(DIRTY_TRAITS, k=rnd.choice([2, 3]))
    traits_str = "\n".join(f"- {t}" for t in traits)
    user = USER_TMPL.format(industry=industry, degree=degree,
                            seniority=seniority, traits=traits_str)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYS_PROMPT},
                          {"role": "user", "content": user}],
                max_tokens=4000,   # reasoner 需要给思考留空间
            )
            data = parse_json_loose(resp.choices[0].message.content)
            text, label = data.get("resume_text"), data.get("label_json")
            if not isinstance(text, str) or not isinstance(label, dict):
                raise ValueError("返回缺 resume_text/label_json")
            if isinstance(label.get("work_years"), float):
                label["work_years"] = int(label["work_years"])
            for e in label.get("education", []):
                if isinstance(e, dict) and "degree" in e:
                    e["degree"] = normalize_degree(e["degree"])
            ok, why = validate_label(label)
            if not ok:
                raise ValueError(f"schema 不合格: {why}")
            return {"resume_text": text.strip(), "label_json": label,
                    "_traits": traits}
        except Exception as e:
            if attempt == 2:
                return {"_error": f"{type(e).__name__}: {e}"}
            time.sleep(1.5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    client = make_client()

    print(f"开始生成 {args.n} 条 OOD 样本 (model={MODEL}, 并发={args.workers}) ...")
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
                print(f"  失败: {r.get('_error', '?')[:120]}")
            if done % 10 == 0 or done == args.n:
                print(f"  进度 {done}/{args.n} | 成功 {len(samples)} | 失败 {errors}")

    # 去重
    seen, uniq = set(), []
    for s in samples:
        if s["resume_text"] not in seen:
            seen.add(s["resume_text"]); uniq.append(s)

    out_path = os.path.join(DATA_DIR, "resume_ood.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps({"resume_text": r["resume_text"],
                                "label_json": r["label_json"]},
                               ensure_ascii=False) + "\n")

    print(f"\n完成: 有效 {len(uniq)} 条 (去重后) | 失败 {errors} -> {out_path}")
    print("脏特征使用统计:")
    from collections import Counter
    cnt = Counter(t for r in uniq for t in r["_traits"])
    for t, c in cnt.most_common():
        print(f"  {c:>2} × {t[:40]}")


if __name__ == "__main__":
    main()
