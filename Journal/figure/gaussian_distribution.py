import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.axisartist as axisartist
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)

fig = plt.figure(figsize=(2, 2))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
#将绘图区对象添加到画布中
fig.add_axes(ax)
mean, sigma = 0, 1
x= np.linspace(mean - 6*sigma, mean + 6*sigma, 100)

y = normal_distribution(x, mean, sigma)
plt.plot(x, y, 'k',linewidth=3, label='m=0,sig=1')

ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("-|>", size = 2.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 2.0)
#设置x、y轴上刻度显示方向
ax.axis["x"].set_axis_direction("top")
ax.axis["y"].set_axis_direction("right")
plt.xticks([])
plt.yticks([])
"""
plt.xticks([])
plt.yticks([])
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
"""
#plt.legend()
#plt.grid()
plt.savefig("gaussian_distribution",dpi=200)
plt.show()