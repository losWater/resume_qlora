"""
M2 用的小型中文 NER 数据集 (本地合成, 不依赖外部下载)。
任务范式: 句子(文本) -> 实体列表(JSON), 与简历抽取"文本->JSON"完全一致。
实体类型: 人名 / 地名 / 机构名。

输出: data/ner_train.jsonl, data/ner_val.jsonl
每行: {"text": "...", "entities": [{"entity":"...","type":"..."}, ...]}

运行:  .venv\\Scripts\\python.exe make_ner_data.py
"""
import sys, os, json, random
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 始终相对脚本所在目录, 避免受当前工作目录影响
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

random.seed(42)  # 固定种子, 可复现

PERSONS = ["张伟", "李娜", "王芳", "刘洋", "陈静", "杨磊", "赵敏", "黄强",
           "周婷", "吴昊", "徐丽", "孙鹏", "马超", "朱琳", "胡军"]
PLACES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安",
          "南京", "重庆", "苏州", "天津", "青岛", "厦门", "长沙"]
ORGS = ["清华大学", "阿里巴巴", "腾讯科技", "中国银行", "华为公司",
        "北京大学", "字节跳动", "复旦大学", "招商银行", "百度公司",
        "浙江大学", "京东集团", "工商银行", "小米科技", "中科院"]

# 句子模板: 用占位符 {p}=人名 {l}=地名 {o}=机构名, 标注哪些槽位是实体
TEMPLATES = [
    ("{p}毕业于{o}，目前在{l}工作。", ["p", "o", "l"]),
    ("去年{p}从{l}搬到了{l2}。", ["p", "l", "l2"]),
    ("{o}总部位于{l}，由{p}负责。", ["o", "l", "p"]),
    ("{p}和{p2}一起参加了在{l}举办的会议。", ["p", "p2", "l"]),
    ("{p}是{o}的高级工程师。", ["p", "o"]),
    ("这次{l}之行，{p}拜访了{o}。", ["l", "p", "o"]),
    ("{o}与{o2}在{l}签署了合作协议。", ["o", "o2", "l"]),
    ("{p}出生于{l}，后来定居{l2}。", ["p", "l", "l2"]),
    ("作为{o}的代表，{p}发表了演讲。", ["o", "p"]),
    ("{p}计划下个月去{l}出差。", ["p", "l"]),
]

TYPE_OF = {"p": "人名", "p2": "人名", "l": "地名", "l2": "地名",
           "o": "机构名", "o2": "机构名"}


def gen_one():
    tmpl, slots = random.choice(TEMPLATES)
    chosen = {}
    used_p, used_l, used_o = [], [], []
    for s in slots:
        if s.startswith("p"):
            v = random.choice([x for x in PERSONS if x not in used_p]); used_p.append(v)
        elif s.startswith("l"):
            v = random.choice([x for x in PLACES if x not in used_l]); used_l.append(v)
        else:
            v = random.choice([x for x in ORGS if x not in used_o]); used_o.append(v)
        chosen[s] = v
    text = tmpl.format(**chosen)
    # 实体按在文本中出现的顺序排列
    entities = [{"entity": chosen[s], "type": TYPE_OF[s]} for s in slots]
    return {"text": text, "entities": entities}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    seen = set()
    samples = []
    # 去重生成
    while len(samples) < 270:
        r = gen_one()
        key = r["text"]
        if key in seen:
            continue
        seen.add(key)
        samples.append(r)
    random.shuffle(samples)
    train, val = samples[:240], samples[240:270]

    for name, data in [(os.path.join(DATA_DIR, "ner_train.jsonl"), train),
                       (os.path.join(DATA_DIR, "ner_val.jsonl"), val)]:
        with open(name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写出 {name}: {len(data)} 条")

    print("\n样例:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
