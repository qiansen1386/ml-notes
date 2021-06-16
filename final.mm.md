<!-- Usage: 安装 markmap 插件 https://marketplace.visualstudio.com/items?itemName=gera2ld.markmap-vscode  -->
# ML 4702

## Exam

### Layout

- [comp4702FinalExam2021.pdf](./comp4702FinalExam2021.pdf)
- Timed Assignment: 3hr + 10mins reading + 15-minute submission period
- 没有 flexible window 了，以Exam timetable 标注时间为准

### 数据

- CSV format
- 10+ columns
- 100~9999 rows
- Real world data with missing fields

### 要求

- 可以用任何库，但要 Reference 出处
  - 要解释 method 和 parameters 的作用，不能有 it works but I.d.k. why
    - 包括重要的 optional parameter
- 可以用 WEKA / netica
- 代码不需要太美观，甚至可以不贴代码，重点在结论和分析。
- 不需要考虑太多引用和格式。尽量不要有 spelling 或 grammartical error
- 不需要复述课程概念，所有课程概念都可以直接引用。
  - 使用第三方库要注意联系课程内容，不要太发散。

### Key points

- Pre-processing
  - 缺失值处理NaN
    - 有时序的 sample 可以
      - 按上下文插值（Interpolation）
        - 复制上一列的信息(Last observation carry forward)
        - 根据前后两列求平均数
      - 用已知数据估计一个Regression模型然后算未知的部分（Extrapolation）
    - 数值类型的可以插中位数或平均数。
      - replacement by mean(更关心平均)
      - replacement by median(outlier 多)
    - 通过跟其他列的相关性逆推（Extrapolation）
  - Remove duplicates
  - Remove Outliers
    - 也可以通过加强bias的方式减小outlier对模型的影响。比如选用强模型
  - lebel数值化
    - Integer Encoding
      - Unique label => an unique integer.
    - One Hot Encoding
      - Unique label => binary column(1 or 0)
  - feature scaling(Normalize)
  - Shuffle
- Training
- Testing
  - Evaluation & Performance
- Visualization


## Theory (TBD)

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

### EM algo (TBD)
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
  - Prac 2 - Q4
  - Clustering model similar to K-means
  - Params: $\mu$ & $\sigma$
  - HyperP: # of compunents

#### Useful Code
```python
sklearn.mixture.GaussianMixture
```

### Density Estimator(TBD)

- Kernel Density Estimitor(Parametric classification)
  - Prac 3
  - non-parametric?
  - HP: Bandwith(smoothing)
- Mixture Models


### Clustering(TBD)

- K-Mean
  - Prac 4
  - K-means clustering and EM with a diagonal covariance matrix with equal terms ( equal circles ) are very similar. You can think of K-Means as removing the variance terms to simplify the EM.
- Hieracical(dendrogram)
- Mean-shift
- 
#### Useful Code

```python
from scipy.cluster import hierarchy
links = hierarchy.linkage(points, method='single', metric='cityblock')
dn = hierarchy.dendrogram(links, labels=labels)
```

### Dimensionality Reduction(TBD)

- PCA
  - Prac 5
- LDA
- t-SNE
  - Prac 5


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
[StatQuest: Random Forests Part](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
 - `gini`
 - Homework 9 & Prac 

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

TO be done

### Models

```python
```

### Metrics

```python
sklearn.metrics.roc_auc_score
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
plt.hist(bins[:-1], bins, weights=counts)
# Image should be a numeric matrix
plt.imshow(image_matrix)
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
# concat and stacking
vstack
hstack
np.concat
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