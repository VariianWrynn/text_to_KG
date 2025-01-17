import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 数据
x = np.linspace(0, 10, 100)
y_line = np.sin(x)
y_bar = np.abs(np.cos(x))
y_scatter = np.sin(x) + 0.5 * np.random.randn(100)

# 创建子图布局
plt.figure(figsize=(12, 8))

# 折线图
plt.subplot(3, 1, 1)
plt.plot(x, y_line, label="sin(x)", linewidth=2)
plt.title("折线图 - Sine Function")
plt.xlabel("X 值")
plt.ylabel("Y 值")
plt.legend()
plt.grid(True)

# 调整布局
plt.tight_layout()
plt.savefig("test.png", dpi=300)	
plt.show()
