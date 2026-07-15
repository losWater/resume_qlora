"""
P1: 推理性能基准。对比两种部署形态:
  A) quant : 4-bit NF4 基座 + LoRA adapter (训练/演示时的形态, 省显存但每层要反量化)
  B) merged: adapter 合并进 bf16 基座 (部署常用形态, 显存翻倍但没有反量化开销)

指标 (29 条 val 真实 prompt, 贪心解码, 与评估口径一致):
  - 首 token 延迟 (prefill): 均值 / P95
  - 端到端延迟: 均值 / P95
  - 解码吞吐: 生成 tokens / 解码时间 (扣除 prefill)
  - 显存: 模型加载后 / 推理峰值 (torch allocated)

两种形态各跑一个独立子进程 (本脚本以 --mode 自调用), 保证显存统计干净。

运行:  .venv\\Scripts\\python.exe bench_infer.py
产物:  outputs/bench_infer_result.json
"""
import sys, os, json, time, subprocess, statistics
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.path.join(HERE, "outputs", "resume_adapter")
VAL_FILE = os.path.join(HERE, "data", "resume_val.jsonl")
OUT_FILE = os.path.join(HERE, "outputs", "bench_infer_result.json")
WARMUP = 2          # 前 2 条热身 (CUDA 内核编译/缓存), 不计入统计
MAX_NEW = 512


def load_model(mode):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    if mode == "quant":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    else:  # merged: bf16 基座 + adapter 合并, 无量化
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model = model.merge_and_unload()
    model.eval()
    return model


def run_mode(mode):
    import torch
    from transformers import AutoTokenizer
    from evaluate import build_prompt

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(VAL_FILE, encoding="utf-8")]

    t0 = time.time()
    model = load_model(mode)
    load_seconds = time.time() - t0
    torch.cuda.synchronize()
    mem_after_load = torch.cuda.memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    first_tok, e2e, gen_tokens, decode_time = [], [], [], []
    for i, r in enumerate(rows):
        text = build_prompt(tok, r["resume_text"])
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # 首 token 延迟 ~= prefill: 只生成 1 个 token
            torch.cuda.synchronize(); t1 = time.time()
            model.generate(**inputs, max_new_tokens=1, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
            torch.cuda.synchronize(); ft = time.time() - t1
            # 端到端: 完整生成 (与评估同参数)
            torch.cuda.synchronize(); t2 = time.time()
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            torch.cuda.synchronize(); ee = time.time() - t2
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        if i >= WARMUP:
            first_tok.append(ft); e2e.append(ee)
            gen_tokens.append(n_new); decode_time.append(ee - ft)
        if (i + 1) % 10 == 0:
            print(f"  [{mode}] 进度 {i+1}/{len(rows)}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    def p95(xs):
        return sorted(xs)[int(0.95 * (len(xs) - 1))]

    result = {
        "mode": mode,
        "n_samples": len(first_tok),
        "max_new_tokens": MAX_NEW,
        "load_seconds": round(load_seconds, 1),
        "vram_model_gb": round(mem_after_load, 2),
        "vram_peak_infer_gb": round(peak_mem, 2),
        "first_token_ms_mean": round(statistics.mean(first_tok) * 1000, 1),
        "first_token_ms_p95": round(p95(first_tok) * 1000, 1),
        "e2e_s_mean": round(statistics.mean(e2e), 2),
        "e2e_s_p95": round(p95(e2e), 2),
        "gen_tokens_mean": round(statistics.mean(gen_tokens), 1),
        "decode_tokens_per_s": round(sum(gen_tokens) / sum(decode_time), 1),
    }
    with open(os.path.join(HERE, "outputs", f"bench_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[{mode}] 完成: 首token {result['first_token_ms_mean']}ms, "
          f"吞吐 {result['decode_tokens_per_s']} tok/s, 峰值显存 {result['vram_peak_infer_gb']}GB")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--mode":
        run_mode(sys.argv[2])
        return

    results = {}
    for mode in ["quant", "merged"]:
        part = os.path.join(HERE, "outputs", f"bench_{mode}.json")
        if not os.path.exists(part):
            print(f"\n===== 测试 {mode} =====")
            ret = subprocess.run([sys.executable, os.path.abspath(__file__), "--mode", mode])
            if ret.returncode != 0:
                print(f"[{mode}] 失败, 中止"); sys.exit(1)
        results[mode] = json.load(open(part, encoding="utf-8"))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"note": "29 条 val 真实 prompt, 贪心解码, 前 2 条热身不计入; "
                           "quant=4bit+adapter, merged=合并后 bf16",
                   "results": results}, f, ensure_ascii=False, indent=2)

    q, m = results["quant"], results["merged"]
    print("\n" + "=" * 70)
    print(f"{'指标':<22}{'4bit+adapter':>16}{'merged bf16':>16}")
    print("-" * 70)
    for label, key, unit in [
        ("模型显存", "vram_model_gb", "GB"),
        ("推理峰值显存", "vram_peak_infer_gb", "GB"),
        ("首token延迟(均值)", "first_token_ms_mean", "ms"),
        ("首token延迟(P95)", "first_token_ms_p95", "ms"),
        ("端到端(均值)", "e2e_s_mean", "s"),
        ("端到端(P95)", "e2e_s_p95", "s"),
        ("解码吞吐", "decode_tokens_per_s", "tok/s"),
    ]:
        print(f"{label:<22}{q[key]:>14}{unit}{m[key]:>14}{unit}")
    print(f"\n结果已保存: {OUT_FILE}")


if __name__ == "__main__":
    main()
