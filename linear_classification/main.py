import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
data = pd.read_csv('/Users/luoyaohui/Desktop/data.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 添加偏置值
X = np.c_[np.ones((X.shape[0], 1)), X]

# 计算参数
w = np.linalg.inv(X.T @ X) @ X.T @ y

#预测标签
y_pred = np.round(X @ w)

#计算准确度
# accuracy = np.mean(y == y_pred) * 100
# print('Accuracy: %.2f%%' % accuracy)
accuracy = accuracy_score(y, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# 打印参数
print(w)

# 绘制三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
colors = np.where(y == 1, 'r', 'b')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, marker='o')

# 计算超平面的位置和方向
w1, w2, w3, b = w
xx, yy = np.meshgrid(range(-2, 3), range(-2, 3))
z = (-w1 * xx - w2 * yy - b) / w3

# 绘制超平面
ax.plot_surface(xx, yy, z, alpha=0.2)

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()