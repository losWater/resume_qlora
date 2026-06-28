"""
环境验收脚本 (里程碑 M1)
目标: 确认 RTX 5060 能被 PyTorch 识别, 且能用 4-bit 量化加载模型不报错。

运行:  .venv\\Scripts\\python.exe check_env.py
"""
import sys

# Windows 控制台默认 GBK，强制 UTF-8 避免打印中文/符号时崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("=" * 60)
print("1. 基本信息")
print("=" * 60)
print(f"Python  : {sys.version.split()[0]}")
print(f"可执行文件: {sys.executable}")

print("\n" + "=" * 60)
print("2. PyTorch + CUDA + 显卡")
print("=" * 60)
try:
    import torch
    print(f"torch 版本      : {torch.__version__}")
    print(f"torch 编译的 CUDA: {torch.version.cuda}")
    avail = torch.cuda.is_available()
    print(f"CUDA 可用       : {avail}")
    if avail:
        print(f"显卡名称        : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"算力 (sm)       : sm_{cap[0]}{cap[1]}  (Blackwell 5060 应为 sm_120)")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存总量        : {total:.1f} GB")
    else:
        print("[失败] CUDA 不可用！很可能 torch 不是 cu128 版本，认不出 50 系卡。")
        print("   重装: pip install torch --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)
except Exception as e:
    print(f"[失败] torch 导入/检测失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("3. 关键库版本")
print("=" * 60)
for mod in ["transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]:
    try:
        m = __import__(mod)
        print(f"{mod:15s}: {getattr(m, '__version__', '未知')}")
    except Exception as e:
        print(f"{mod:15s}: [失败] 导入失败 {e}")

print("\n" + "=" * 60)
print("4. 4-bit 量化冒烟测试 (加载一个极小模型)")
print("=" * 60)
print("说明: 验证 bitsandbytes 能在 5060 上跑 4-bit 加载。用最小模型，只测加载不训练。")
try:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "sshleifer/tiny-gpt2"  # 极小模型, 仅验证 4-bit 链路, 不需要联网下大模型
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"尝试 4-bit 加载: {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="cuda"
    )
    dev = next(model.parameters()).device
    print(f"[成功] 4-bit 加载成功！模型在: {dev}")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"[警告] 4-bit 加载失败（可能是网络下不到测试模型，或 bnb 不兼容）:")
    print(f"   {type(e).__name__}: {e}")
    print("   若是网络问题可忽略；若是 bitsandbytes/CUDA 报错，把以上原文发给 Claude Code 诊断。")

print("\n" + "=" * 60)
print("验收完成。若第 2 节 CUDA 可用=True 且显卡为 5060，环境基本就绪。")
print("=" * 60)
