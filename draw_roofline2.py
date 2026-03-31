import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 硬件理论参数设定 (Nvidia RTX 3070)
# ==========================================
PEAK_BANDWIDTH = 448.0   # 理论峰值显存带宽 (GB/s)
PEAK_FLOPS = 20300.0     # 理论峰值算力 (GFLOPS)
RIDGE_POINT = PEAK_FLOPS / PEAK_BANDWIDTH  # 拐点

# ==========================================
# 2. 实测模型数据点 (TinyLlama-1.1B 实测推导)
# ==========================================
# Decode 阶段:
decode_i = 1.0        # 算术强度 I (FLOPs/Byte)，逐字生成依然极低
decode_flops = 46.0   # 实际算力表现飙升至 46 GFLOPS

# Prefill 阶段:
prefill_i = 20.0      # 算术强度 I (FLOPs/Byte)，因为计算量大，强度更高
prefill_flops = 1653.0 # 实际算力表现暴涨至 1653 GFLOPS

# ==========================================
# 3. 开始绘制 Roofline 图形
# ==========================================
plt.figure(figsize=(10, 7), dpi=150)
x = np.logspace(-1, 3, 500)

# 计算 Roofline 边界并绘制
y_roof = np.minimum(PEAK_FLOPS, PEAK_BANDWIDTH * x)
plt.plot(x, y_roof, color='black', linewidth=2.5, label='RTX 3070 Theoretical Roofline')

# 划分瓶颈区域
plt.axvline(x=RIDGE_POINT, color='gray', linestyle='--', alpha=0.7)
plt.fill_between(x, 0, y_roof, where=(x < RIDGE_POINT), color='blue', alpha=0.05, label='Memory Bound Region')
plt.fill_between(x, 0, y_roof, where=(x >= RIDGE_POINT), color='red', alpha=0.05, label='Compute Bound Region')

# 绘制 TinyLlama 实测数据点 (换成醒目的绿色和紫色)
plt.scatter(decode_i, decode_flops, color='cyan', s=150, zorder=5, edgecolors='black', label='TinyLlama Decode')
plt.scatter(prefill_i, prefill_flops, color='magenta', s=150, zorder=5, edgecolors='black', label='TinyLlama Prefill')

# 添加文本注释
plt.annotate(f'Decode\n({decode_i}, {decode_flops:.0f} GFLOPS)', 
             xy=(decode_i, decode_flops), xytext=(decode_i*0.5, decode_flops*2.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=11)

plt.annotate(f'Prefill\n({prefill_i}, {prefill_flops:.0f} GFLOPS)', 
             xy=(prefill_i, prefill_flops), xytext=(prefill_i*0.5, prefill_flops*2.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=11)

# 设置对数坐标轴
plt.xscale('log', base=10)
plt.yscale('log', base=10)

# 图表美化与标签
plt.title('Roofline Model: TinyLlama-1.1B on RTX 3070 (High Freq)', fontsize=16, fontweight='bold')
plt.xlabel('Arithmetic Intensity [FLOPs/Byte]', fontsize=12)
plt.ylabel('Performance [GFLOPS]', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='upper left', fontsize=11)

# 限制坐标轴显示范围
plt.xlim(10**-1, 10**3)
plt.ylim(10**0, 10**5)

# 显示并保存图像
plt.tight_layout()
plt.savefig('rtx3070_roofline_tinyllama.png')
plt.show()