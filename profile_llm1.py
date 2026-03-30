import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity

model_id = "Qwen/Qwen1.5-0.5B"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# 明确指定放在第 0 张卡上
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)

prompt = "Explain the roofline model in computer architecture:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Warming up GPU...")
# 先空跑一次预热 GPU，让显存分配完毕，测出来的时间才准
with torch.no_grad():
    model(**inputs)

print("Starting PyTorch Profiler...")
# 启动 PyTorch 探针！开启 flops 计算和 shape 记录
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True  # 核心参数：让 PyTorch 帮我们算 FLOPs
) as prof:

    # --- 1. 抓取 Prefill 阶段 ---
    with record_function("Stage_Prefill"):
        with torch.no_grad():
            outputs = model(**inputs)

    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)

    # --- 2. 抓取 Decode 阶段 ---
    with record_function("Stage_Decode"):
        with torch.no_grad():
            # 生成 5 个 Token 测一下平均水平
            for _ in range(5):
                out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                next_token = torch.argmax(out.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)

# 打印出最耗时的前 15 个算子
print("\n=== Profiling Results ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# 导出一个可视化的 json 文件，我们可以拖到浏览器里看时间线
prof.export_chrome_trace("llm_trace.json")
print("\nTrace saved to llm_trace.json")