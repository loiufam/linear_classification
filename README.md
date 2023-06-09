# linear_classification
用python实现最小二乘法来设计线性分类器

一、实验目的和要求
实验目的：本实验旨在通过编写Python程序，运用最小二乘法法设计线性分类器，通过权重向量来对新的数据进行预测，并通过计算准确率来评估模型的性能，以掌握线性回归的基本原理和实现方法。
实验要求：
1.熟悉最小二乘法的基本原理和公式；
2.将数据集分为训练集和测试集，用训练集训练模型，用测试集评估模型的性能。
3.编写Python程序，实现最小二乘法训练一个线性分类器的算法；
4.使用给定的数据集进行训练，并计算出模型的预测误差；
5.对实验结果进行分析和讨论，包括模型拟合程度、误差分析等方面。


二、实验内容与原理
实验内容：
实验步骤包括：
1.加载数据集
首先，需要使用Python的pandas库加载csv数据集，将数据集分为特征向量和标签向量，并将特征向量添加偏置值，构成增广特征向量。
2.计算权重向量
接着，可以使用最小二乘法计算权重向量w，用来预测新的数据。权重向量的计算公式为：w = inv(X.T*X)X.Ty，其中X为增广特征向量，y为标签向量，inv表示矩阵的逆。
3.预测标签
使用计算出的权重向量w对新的数据进行预测，并将预测值四舍五入得到0或1的标签。
4.计算准确率
最后，计算预测准确率来评估模型性能。预测准确率可以使用numpy的mean函数计算，并将其乘以100以显示百分比。
实验原理：
线性分类器是一种用于二分类问题的机器学习算法，其中数据被分为两个类别，分别用0和1表示。线性分类器的目标是找到一个超平面，将两个类别分开。超平面可以表示为一个线性方程：w0x0 + w1x1 + ... + wnxn = 0，其中x是特征向量，w是权重向量，w0是偏置值。在二维空间中，超平面可以表示为一条直线。
最小二乘法可以用于计算权重向量w，使得预测值与实际标签之间的误差最小化。最小二乘法的原理是，找到使误差平方和最小的权重向量w。这可以通过求解一个线性方程组来实现。用一下公式可以计算权重向量w：
w = inv(X.T*X)X.Ty
其中，X.T表示X的转置矩阵，inv表示逆矩阵。计算出的权重向量可以用来预测新的数据。
最后将实际标签向量y和预测标签向量y_pred逐一进行比较，计算它们相等的比例，乘以100得到百分比，即为模型预测准确率。准确率越高，则表示模型性能越好。

三、实验环境
1．硬件环境 CPU：mac m1；RAM：8G；硬盘：256G
2．软件环境 MacOS Ventura 13.1；python 3.9


五、实验记录（系统测试）
如果是编程实验，则为系统测试，提供测试输入数据、结果数据或状态，并给出若干个（至少两个）不同的输入和结果。
输入数据：实验数据data.csv，有100个样本；

用pandas库的函数读取所有数据集样本，用于训练模型。


结果数据：

输出值为准确率和权重向量：
 
3D图像显示可视化结果：


六、实验结果与分析
根据得到的结果，可以得到线性分类器为：
y = 0.1635X1 + 0.0487X2 - 0.0003X3  - 0.1822。
要想让模型预测更准确，要精量减小预测值和真实值的误差，可以考虑梯度下降的算法获取最优的参数。
如果预测准确率很低，我们需要重新考虑选择哪些特征，或者使用其他的机器学习算法来解决问题。如果预测准确率较高，我们可以使用该模型对新的数据进行预测。同时，我们可以通过分析权重向量来了解哪些特征对预测最重要，这可以为后续的特征工程提供指导。




七、实验心得与建议
在预测准确率时，我发现模型的准确率计算不精确，所以引入了sklearn.metrics库的accuracy_score来预测准确率。
在进行线性分类器的实验中，我深刻地认识到了机器学习的重要性和复杂性。通过对数据集进行预处理需要考虑特征的相关性、噪声和缺失值等因素。此外，选择合适的机器学习算法和超参数也非常重要。
在实验中，我发现使用最小二乘法训练线性分类器的过程相对简单，但是在实际应用中，该方法可能会受到数据的噪声和异常值的影响，导致模型表现不佳。因此，在实际应用中，需要使用更加鲁棒的方法来解决问题。
最后，我认为在进行机器学习实验时，不仅要注重理论知识的掌握，还需要结合实际问题，通过不断实践来提高自己的技能。同时，需要保持学习的态度，及时关注最新的研究进展和技术趋势，以不断提高自己的竞争力。
