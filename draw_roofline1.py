import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 硬件理论参数设定 (Nvidia RTX 3070)
# ==========================================
# 理论峰值内存带宽 (GB/s)
# 3070 显存位宽 256-bit, 显存频率 14Gbps -> 448 GB/s
PEAK_BANDWIDTH = 448.0

# 理论峰值算力 (GFLOPS)
# 3070 FP32 单精度理论峰值约为 20.3 TFLOPS -> 20300 GFLOPS
# (若完全跑满 FP16 Tensor Core 会更高，这里以基准 FP32 算力作为参考演示)
PEAK_FLOPS = 20300.0

# 计算拐点 (Ridge Point): 算力天花板与带宽天花板的交点
RIDGE_POINT = PEAK_FLOPS / PEAK_BANDWIDTH

# ==========================================
# 2. 实测模型数据点 (Qwen1.5-0.5B 实测推导)
# ==========================================
# Decode 阶段: 极低的算术强度，受制于显存读取
decode_i = 1.0        # 算术强度 I (FLOPs/Byte)
decode_flops = 23.0   # 实际算力 (GFLOPS)

# Prefill 阶段: 算术强度随 Prompt 长度增加
prefill_i = 10.0      # 算术强度 I (FLOPs/Byte)
prefill_flops = 234.0 # 实际算力 (GFLOPS)

# ==========================================
# 3. 开始绘制 Roofline 图形
# ==========================================
plt.figure(figsize=(10, 7), dpi=150)

# 生成 X 轴数据 (算术强度，使用对数刻度范围)
x = np.logspace(-1, 3, 500)

# 计算 Roofline 的两条边界
# Y = min(Peak_FLOPS, Bandwidth * I)
y_roof = np.minimum(PEAK_FLOPS, PEAK_BANDWIDTH * x)

# 绘制天花板曲线
plt.plot(x, y_roof, color='black', linewidth=2.5, label='RTX 3070 Theoretical Roofline')

# 划分 Memory Bound 和 Compute Bound 区域
plt.axvline(x=RIDGE_POINT, color='gray', linestyle='--', alpha=0.7)
plt.fill_between(x, 0, y_roof, where=(x < RIDGE_POINT), color='blue', alpha=0.05, label='Memory Bound Region')
plt.fill_between(x, 0, y_roof, where=(x >= RIDGE_POINT), color='red', alpha=0.05, label='Compute Bound Region')

# 绘制实测数据点
plt.scatter(decode_i, decode_flops, color='blue', s=150, zorder=5, edgecolors='black', label='Stage: Decode')
plt.scatter(prefill_i, prefill_flops, color='red', s=150, zorder=5, edgecolors='black', label='Stage: Prefill')

# 添加文本注释
plt.annotate(f'Decode\n({decode_i}, {decode_flops:.0f} GFLOPS)',
             xy=(decode_i, decode_flops), xytext=(decode_i*0.5, decode_flops*2.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=11)

plt.annotate(f'Prefill\n({prefill_i}, {prefill_flops:.0f} GFLOPS)',
             xy=(prefill_i, prefill_flops), xytext=(prefill_i*0.5, prefill_flops*2.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=11)

# 设置对数坐标轴 (Roofline 标配)
plt.xscale('log', base=10)
plt.yscale('log', base=10)

# 图表美化与标签
plt.title('Roofline Model: Qwen-0.5B on RTX 3070', fontsize=16, fontweight='bold')
plt.xlabel('Arithmetic Intensity [FLOPs/Byte]', fontsize=12)
plt.ylabel('Performance [GFLOPS]', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='upper left', fontsize=11)

# 限制坐标轴显示范围，让图形居中好看
plt.xlim(10**-1, 10**3)
plt.ylim(10**0, 10**5)

# 显示并保存图像
plt.tight_layout()
plt.savefig('rtx3070_roofline_llm.png')
plt.show()