<!-- Usage: 安装 markmap 插件 https://marketplace.visualstudio.com/items?itemName=gera2ld.markmap-vscode  -->
# ML 4702

## Theory

### Supervised Learning Principles
- Classification vs Regression
- Generalization & Model complexity
  - Underfit
    - The complexity of the model is lower than data
    - Loss on both testing /training set is high
  - Overfit
    - Model can perfectly fit training data as well as noises
    - Loss on training set is low, but not true for testing set
  - To minimize overfitting => Check "**Lowest** validation error"(NOT "elbow")
- Regularization
  - Add a penalty term into loss function to limit the complexity of the model and decrease overfitting.(Pull/smooth out a curve towards a linear function)
- Metrics
  - Regression
    - (unadjusted)**R-Squared** $R^2= Explained\ variation / Total\ variation$
    - **SSE** sum of squared error $SSE(y,\hat{y})=\sum^n_i{(y_i-\hat{y_i})^2}$
    - **RMSE** Root Mean Squared Error$RMSE(y,\hat{y})=\sqrt{\frac{1}{n}\sum^n_i{(y_i-\hat{y_i})^2}}$
      - **unbiased** RMSE use **df**(degree of freedom) instead of **N**
  - Classification
    - Confusion matrix
    - $Accuracy = (TP+TN)/(TP+FP+FN+TN)$
      - If classification problems is well balanced and not skewed or No class imbalance
      - 反例: 被车撞的概率，一直说No也有99%的正确率。
    - $Precision = (TP)/(TP+FP)$
      - 所有我们标示为阳性的记录里有多少是真阳性
      - If we want to be very sure about what we say, like creditcard fraud detection -> false alert leads to customer complaints
      - Caveat: High precision = Low sensitivity
      - Related: $Specificity = (TN)/(TN+FP)$
        - 所有阴性判断中有多少是真的
    - $Recall = (TP)/(TP+FN)$ 
      - 所有实际为阳性的记录中有多少被我们识别了
      - If we want to capture as many positive as possible.
      - 如果我盲目返回 Positive，Recall 会是 100%（某种程度上反应 Sensitivity，需要和Precision 平衡一下）
    - $F_1=2\times\frac{precision\times recall}{precision + recall}$
      - (用来平衡precision和sensitivity)
    - ROC curve
      - Compare the True Positive Rate vs False Positive Rate($Sensitivity$ vs $1-Specificity$)
      - 用来对比和决定**最佳阈值**应该设置在哪里，一个 ROC 曲线中的每一个点都有一个F score。
      - AUC相当于对这些 F 取了平均。所以如果只是用来衡量算法本身的好坏最好用 $F_1$ score
- Combining Learner (TBD)
  - Bagging and Boosting (TBD)
  - Prac 9 


## Algorithm

### Polynomial(Linear) Regression

- Prac 2-Q1~Q3
- Params: Polynomial coefficients
- HP: poly_degree
- Related
  - Logistic Regression
  

#### Useful Code
```python
np.polyfit()
np.polyval()
scipy.optimization.curve_fit
sklearn.matrics.r2_score

sklearn.preprocessing.PolynomialFeatures
```

### EM algo
- [Expectation Maximization](https://zhuanlan.zhihu.com/p/40991784)
  - E-step
    - Train the model parameters
  - M-step
    - Generate new fields/clustering w/ new model
  - Until it converged
    - (all missing fields are filled or clustering become stablized)
- KNN (EM)
  - non-parametric
- Parametric Probabilistic Classification
  - Gaussian Mixture Model (GMM)尝试找出一个模型（通常为联合正态分布）和对应的 parameters 来描述我们已有的模型
  - Prac 2 - Q4
  - Clustering model similar to K-means
  - Params: $\mu$ & $\sigma$
  - HyperP: # of compunents
- Kernel Density Estimitor(Parametric classification)
  - 有点像histogram只不过累积的不是小方格而是Kernel function。组合起来可能太曲里拐弯的，所以需要熨平一点smoothing。
  - Prac 3
  - non-parametric
  - HP: Bandwith(smoothing)


#### Useful Code
```python
sklearn.mixture.GaussianMixture
```


### Clustering

- K-Mean
  - Prac 4
  - K-means clustering and EM with a diagonal covariance matrix with equal terms ( equal circles ) are very similar. You can think of K-Means as removing the variance terms to simplify the EM.
  - K-Means有两个缺点
    - 需要**先知道k值**，如果k值取得不好，效果可能很差。
    - k-means收敛到局部最优，所以这个算法优化结果**依赖于初始点选取**
  - 因为K-means计算点点之间的欧氏距离，所以跟RBF核颇有渊源。
- Mean-shift
  - 是已知聚类标签的情况下寻找最佳中心点的算法[知乎详情](https://zhuanlan.zhihu.com/p/31183313)
    - 相当于先通过kernel函数一般是RBF进行图像处理（理解为blur）
    - 然后在blur之后的图像上进行梯度上升Gradient Ascend。
- Hieracical(dendrogram)


#### Useful Code

```python
from scipy.cluster import hierarchy
links = hierarchy.linkage(points, method='single', metric='cityblock')
dn = hierarchy.dendrogram(links, labels=labels)
```

### Dimensionality Reduction

- PCA
  - Prac 5
- LDA(TBD)
- [t-SNE](https://zhuanlan.zhihu.com/p/103261749)
  - t-SNE的降维关键：把高纬度的数据点之间的距离转化为高斯分布概率。
  - 高纬度相似度用高斯，低纬度用t分布，然后设置一个惩罚函数（KL散度(Kullback-Leibler divergence)），就实现了x降低维度但是保留一定局部特征的方法。
  - Prac 5
  - 与SNE的区别，通过t分布的长尾特性，解决了传统SNE的拥挤问题。
  - Parameters
    - n_components 低维空间的维度
    - perpexity 混乱度，优化过程中考虑的临近点的数量，一般选30
    - early_exaggeration 表示嵌入空间簇间距的大小，默认为12，越大簇和簇之间的间距越大 
    - learning_rate 学习率，控制梯度下降的快慢默认200(100~1000)
    - n_iter 迭代次数默认1000 (>250)
    - init `random` 或`pca` 是否先用PCA初始化，也可以提供数组(n_samples, n_target)
    - method 优化方法，默认用`barnets_hut`一种模拟算法O(NlogN)，也可以选`exact`计算标准的KL散度O(N^2)，误差小
    - random_state 随机数种子，因为loss函数不是凸函数，存在局部最佳，不同的随机函数会产生不同的结果。可以多次尝试选择最小的loss


### - NeuralNet

- MLP
  - Prac 6
- CNN
  - Prac7

### SVM
- Prac8
  - Weka
  - HP: 一般用 grid-search 暴力尝试不同组合
    - C -> controls the regulation term(How many error can we tolerate, controls the margin in an indirect manner)
      - $\in [2^{-5},2^{15} ]$
    - Kernel:
      - RBF: gamma-> $\in [2^{-5},2^3]$
      - Poly: d -> degree (0~8)

### Decision Tree & Random Forest
- Homework 9 & Prac 
[StatQuest: Random Forests Part](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- `gini`
  - To measure the impurity
- Involved into 
 - [Random Forest](https://youtu.be/J4Wdy0Wc_xQ?t=14)
 - [Adaboost](https://www.youtube.com/watch?v=LsK-xG1cLYA)
 - Gradient Boosting
   - xgboost
- 一般没有直接用决策树的，都是集成学习(Ensemble Learning)的一部分
  - 三个臭皮匠赛过诸葛亮。需要每个子决策树尽可能准(accurate)，但尽可能不一样(diverse)
  - 集成学习主要是有几种方法bagging, boosting, stacking等
    - bagging - Random forest 通过boostraping（数据集小）或随机分组（数据集大）训练不一样的完整子树，然后投票，综合意见
    - boosting - Adaboost 通过训练一堆二叉弱分类器(stump)，每个stump有自己的weight（不是一人一票了）。

### Graphical Model
- Benefit:
  - **Transparent**, provide reasoning
  - **Efficient** algorithm exist -> a chain of bayes rules
- BayesianNet
  - Prac 10
    - Netica
  - We assume some of the variables are independent, some are dependent (shown as edges)

### Bayesian Inference
- Gaussian Process Regression
  - https://zhuanlan.zhihu.com/p/60987749
  - https://zhuanlan.zhihu.com/p/75589452
  - Prac 11
  - Non-parametic
  - HP: Distance Kernel
    - RBF Kernel: l, N
      - l 越大，图像越平滑
    - Periodic Kernel, etc. see [The Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/)
  - Steps: 
    1. Define a kernel to compute distance matrix
    2. Sample a prior
    3. Update the prior according to data
    4. Plot the posterior
       - new **mean** is the predicted function
       - new **cov_mtx** can be used to draw the confident interval on every point of `t`

```python
scipy.spatial.distance.cdist(p, q,'sqeuclidean')
from sklearn.metrics.pairwise import rbf_kernel
np.random.default_rng().multivariate_normal(mean, cov, size=5)
plt.fill_between(linspace, mean + post_std, mean - post_std, alpha = 0.5)
```

## Python

### Imports

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

### Data Loading

pandas is the fastest
np.loadtxt is most basic
np.genfromtxt is safer when it comes to missing data.
There is also `import csv`, but too tedius to use

```python
df=pd.read_csv('myfile.csv', sep=',',header=None)
print(df.values)
# array([[ 1. ,  2. ,  3. ],
#        [ 4. ,  5.5,  6. ]])
arr=np.loadtxt('myfile.csv',delimiter=',')
arr=np.genfromtxt('myfile.csv',delimiter=',')
# array([[ 1. ,  2. ,  3. ],
#        [ 4. ,  5.5,  6. ]])
```

### Pre-processing

#### Spot NaN
```python
# if contains NaN in an array
s = pd.Series([2,3,np.nan,7,"The Hobbit"])
s.isnull().values.any()
# if contains NaN in a DataFrame
df.isnull().sum()
# 0    3 <- meaning column 1 have 3 nulls 
# 1    0
```

#### Check Correlation
```python
corr = dataframe.corr()
corr.style.background_gradient(cmap='coolwarm')

# Use Seaborn
# https://seaborn.pydata.org/generated/seaborn.pairplot.html
import seaborn as sns
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species", height=2.5)
g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", height=2.5)
```

#### imputation
```python
# df is a pandas DataFrame
df.fillna(constant)



```

### Models

```python
```

### Metrics

```python
sklearn.metrics.roc_auc_score
sklearn.metrics.f1_score(y_true, y_pred, labels=None)
sklearn.metrics.r2_score(y_true, y_pred, sample_weight=None) # 1- SSE/SST
```

### Plot

#### subplot
```python
fig, ax = plt.subplots(figsize=(8,6))
fig, ax = plt.subplots(nRows, nCols, figsize=(8,6))
fig = plt.figure(figsize=(3, 3), dpi=120)
ax = fig.add_axes([0, 0, 1, 1])
```

#### drawing

```python
ax.scatter(X, Y,marker='+')
ax.plot(X_line,Y_line,'k',label='Text to show in legend')
ax.vlines(0,-0.1,20.1,linestyles='dotted')
ax.hlines(0,-0.1,20.1,linestyles='solid')
# histogram
counts, bins = np.histogram(data)
# plt.hist(x, bins = number of bins)
plt.hist(bins[:-1], bins, weights=counts)
# Image should be a numeric matrix
plt.imshow(image_matrix)
# show matrix == imshow(matrix, interpolation="none")
plt.matshow(matrix, interpolation="none")
```

#### Labels & axes
```python
plt.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title('Title of subplot')

# fine tune axis
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html
ax.set_ylim((-pi-1,pi+1))
ax.set_xlim((0,5))
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.set_aspect('equal') # square axes
```

#### Save Image

```python
plt.savefig("file.png",dpi=300, format='png',bbox_inches='tight')
plt.show()
```

### Numpy

#### methods
```python
np.ceil()
np.floor()
np.log2(X)
np.std() # => std deviation
np.cov(arr)#=> cov matrix
np.random.normal()
line = np.linspace(-6, 6, num=60)
np.linalg.inv() # inverse of a matrix.
np.dot()
np.matmul() # matrix multiply
```

#### array

##### Create ndarray
```python
np.zeros((r,c), dtype=float)
# np.ones np.eye
np.arange(N).reshape(shape)
np.array(python_list) # python list to ndarray
ndarray.tolist() # ndarray to python list
# 2d
np.arange(100).reshape((-1,2)).shape # (50,2), -1 means auto
```

##### play with arrays
```python
A = np.array([1,2])
B = np.array([3,4])
np.concatenate((A,B))
# array([1, 2, 3, 4])
np.vstack((A,B)) # <=> vsplit()
# array([[1, 2],
#        [3, 4]])
np.hstack((A,B)) # <=> hsplit()
# array([[1, 2, 3, 4]])
```

##### Properties

```python
arr = np.array([[3,4,6], 
                [0,8,1]], dtype = np.int16)
arr.size # 6 # of items
arr.ndim # 2 # of dimension
arr.shape # (2, 3)
arr.itemsize # 2, returns the size (in bytes) of each element of a NumPy array.
arr.dtype #dtype('int16')
```

##### Increase dimension

```python
x = np.array([1, 2])
x.shape
# (2,)
# 4 ways to expand dimensions
np.expand_dims(x, axis=0)
# ==> x[np.newaxis, :] 
np.expand_dims(x, axis=1)
# ==>  x[:, np.newaxis]
# Or Reshape
np.arange(1000).reshape((-1, 2))
```

##### Decrease dimension

```python
# squeeze
x = np.array([[[0], [1], [2]]])
x.shape # (1, 3, 1)
np.squeeze(x).shape # (3,)
np.squeeze(x, axis=0).shape # (3, 1)
np.squeeze(x, axis=1).shape # error : size of selected axis != 1
np.squeeze(x, axis=2).shape # (1, 3)
# flatten 好像是只作用于 2d 数组
a = np.array([[1,2], [3,4]])
a.flatten() # array([1, 2, 3, 4]) ,default is row major
# == a.reshape(-1)
a.flatten(order='F') # array([1, 3, 2, 4]) order = column major
a.reshape(-1,1) # flatten into into column vector 
# == a.reshape(-1)[np.newaxis, :].T 
```