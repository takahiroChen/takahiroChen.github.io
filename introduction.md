---
title: Introduction
layout: template
filename: introduction.md
--- 
# 1. Introduction

## What is machine learning?

> A computer program is said to learn from experience E with respect to some class of tasks T, and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.



## Supervised Learning

### Definition

The task $T$ is to learn a mapping $f$ from inputs $\boldsymbol x\in\cal X$ to $\boldsymbol y\in\cal Y$. The inputs $\boldsymbol x$ is also called the **features**, **covariates**, or **predictors**. If $\boldsymbol x$ is represented by a vector in $\mathbb R^D$, $D$ is the **dimensionality** of the vector. The output $\boldsymbol y$ is called the **label, target** or **response**. The experience $E$ is given in the form of a set of $N$ input-output pairs $\mathcal D = \{(\boldsymbol x_n, \boldsymbol y_n)\}_{n=1}^N$, known as the **training set**.  $N$ is called the **sample size**.

---

### Classification

#### Definition

The output space is a set of $C$ unordered and **mutually exclusive labels** known as **classes**, $\mathcal Y=\{1,2,\cdots,C\}$. The task of predicting the class label is also called **pattern recognition**. If $C=2$, it is called **binary classification**.



#### Tabular Data

If inputs are of fixed size, it is common to store the dataset $\cal D$ into an $N\times D$ matrix, which is known as a **design matrix**. The dataset is an example of **tabular data**. If inputs are of variable size, we can convert them into a fixed-sized representation. This process is also known as **featurization**.



#### Exploratory Data Analysis

It is a good idea to perform **exploratory data analysis** to see if there are any obvious patterns or any obvious problems with the data. **Pair plot** is an example of exploratory data analysis. In a pair plot, panel $(i, j)$ shows a scatter plot of variables $i$ and $j$, and the diagonal entries $(i,i)$ show the marginal density of variable $i$. For higher-dimensional data, it is common to first perform **dimensionality reduction**.

<center>
    <img src = "https://github.com/takahiroChen/blog-img/blob/main/20220529085413.png?raw=true"
         style = "zoom:25%">
    <br>
    Illustration of the Pair Plot
</center>



#### Decision Tree

Decision rules can be arranged into a nested tree structure, called the **decision tree**.



#### Empirical Risk

**Misclassification rate** is defined as:
$$
\mathcal L(\theta) \triangleq \frac{1}{N} \sum_{n=1}^N\mathbb I(y_n \neq f(\boldsymbol x;\boldsymbol \theta)), \tag{1}
$$
where $\mathbb I(e)$ is the binary indicator function, defined as:
$$
\mathbb I(e) \triangleq
\begin{cases}
1 & \text{if }e\text{ is true,}\\
0 & \text{if }e\text{ is false.}
\end{cases}\tag{2}
$$
**Empirical risk** is the average loss of the predictor on the training set:
$$
\mathcal L(\boldsymbol \theta) \triangleq \frac{1}{N} \sum_{n=1}^N \ell(y_n,f(\boldsymbol x;\boldsymbol \theta)),\tag{3}
$$
where $\ell(y,\hat y)$ is the **loss function**. When we use the **zero-one loss**, the empirical risk becomes the misclassification rate.

**Model fitting** or **training** is to find a setting of parameters $\boldsymbol{\hat \theta}$ that minimizes the empirical risk on the training set:
$$
\boldsymbol{\hat \theta}=\underset{\boldsymbol{\theta}}{\text{argmin}}\frac{1}{N} \sum_{n=1}^N \ell(y_n,f(\boldsymbol x;\boldsymbol \theta)).\tag{4}
$$
This is also called **empirical risk minimization**.



#### Uncertainty

Lack of knowledge of input-output mapping is called **epistemic uncertainty** or **model uncertainty**. Intrinsic stochasticity in the mapping is called **aleatoric uncertainty** or **data uncertainty**.

We can capture our uncertainty using **conditional probability distribution**:
$$
p(y=c|\boldsymbol x;\boldsymbol \theta) = f_c(\boldsymbol x;\boldsymbol \theta),\tag{5}
$$
where $f:\mathcal X \to [0,1]^C$ maps inputs to a probability distribution over the $C$ possible output labels. We require $0 \leq f_c \leq 1$ for each $c$ and also $\sum_{c=1}^Cf_c=1$.

It is common to let the model return **unnormalized log-probabilities**, then convert them to probabilities using the **softmax function**:
$$
\mathcal S(\boldsymbol a) \triangleq \left[ \frac{\exp (a_1)}{\sum_{c'=1}^C \exp (a_{c'})},\cdots,  \frac{\exp (a_C)}{\sum_{c'=1}^C \exp (a_{c'})}\right]. \tag{6}
$$
The inputs to the softmax, $\boldsymbol{a} = f(\boldsymbol{x};\boldsymbol{\theta})$ are called **logits**. Then we define
$$
p(y=c|\boldsymbol x;\boldsymbol \theta) = \mathcal S_c(f(\boldsymbol{x};\boldsymbol{\theta})).\tag{7}
$$


#### Logistic Regression

When $f$ is an affine function of the form
$$
f(\boldsymbol{x};\boldsymbol{\theta})=\boldsymbol{w}^\top\boldsymbol{x}+b, \tag{8}
$$
where $\boldsymbol{\theta} = (\boldsymbol w, b)$ are the parameters of the model, the model is called **logistic regression**. In ML, $\boldsymbol w$ are called the **weights** and $b$ is called the **bias**.

It is common to write $\boldsymbol \theta$ as $\tilde{\boldsymbol w}=[b,w_1,\cdots,w_D]$ and define $\tilde{\boldsymbol x}=[1, x_1, \cdots, x_D]$, so that
$$
f(\boldsymbol{x};\boldsymbol{\theta})=\tilde{\boldsymbol{w}}^\top\tilde{\boldsymbol{x}}. \tag{9}
$$


#### Negative Log Likelihood

When fitting probabilistic models, it is common to use the **negative log probability** as our loss function:
$$
\ell (y,f(\boldsymbol x;\boldsymbol \theta))=-\log p(y\vert f(\boldsymbol x;\boldsymbol \theta)). \tag{10}
$$
The **negative log likelihood** is the average negative log probability of the training set:
$$
\text{NLL}(\boldsymbol \theta)=-\frac{1}{N} \sum_{n=1}^N \log p(y\vert f(\boldsymbol x;\boldsymbol \theta)). \tag{11}
$$
We can find the parameters by **maximum likelihood estimate (MSE)**:
$$
\hat{\boldsymbol \theta}_\text{mle}=\underset{\boldsymbol \theta}{\text{argmin}}\,\text{NLL}(\boldsymbol \theta). \tag{12}
$$

---

### Regression

#### Definition

When the output space is the real number: $y\in\mathbb R$, the prediction is called **regression**.



#### L~2~ Loss

In regression, we usually use **quadratic loss**, **or L~2~ loss**:
$$
\ell_2(y, \hat y)=(y-\hat y)^2. \tag{13}
$$
The empirical risk when using L~2~ loss is equal to the **mean squared error (MSE)**:
$$
\text{MSE}(\boldsymbol \theta)=\frac{1}{N}\sum_{n=1}^N(y_n-f(\boldsymbol x;\boldsymbol \theta))^2. \tag{14}
$$

#### Gaussian (Normal) Regression

In regression, it is common to assume the output distribution is a **Gaussian** or **normal**, which is defined as
$$
\mathcal N(y|\mu, \sigma^2) \triangleq \frac{1}{\sqrt{2\pi\sigma^2}}\exp (-\frac{(y-\mu)^2}{2\sigma^2}), \tag{15}
$$
where $\mu$ is the **mean**, $\sigma^2$ is the **variance**, and $\sqrt{2\pi\sigma^2}$ is the **normalization constant** to ensure that the density integrates to 1.

Assuming that $\sigma^2$ is fixed, the NLL becomes
$$
\begin{split}
\text{NLL}(\boldsymbol \theta) &= -\frac{1}{N}\sum_{n=1}^N\log\left[\left(\frac{1}{2\pi\sigma^2}\right)^{1/2} \exp\left(-\frac{1}{2\sigma^2}(y_n-f(\boldsymbol x_n;\boldsymbol \theta))^2\right)\right]\\
& = \frac{1}{2\sigma^2}\text{MSE}(\boldsymbol \theta)+\text{const.}
\end{split}\tag{16}
$$
Computing the MLE of the parameters will result in minimizing the MLE.



#### Linear Regression

If $f(\boldsymbol x;\boldsymbol \theta) = \tilde{\boldsymbol{w}}^\top\tilde{\boldsymbol{x}}$, the regression is called **linear regression**.



#### Polynomial Regression

If the regression has the form $f(x,\boldsymbol w)=\boldsymbol{w}^\top \phi(x)$, where $\phi(x)=[1, x, x^2,\cdots,x^D]$, the regression is called **polynomial regression**.

With the number $D$ increasing, we can perfectly interpolate the data. However, the resulting function will not be a good predictor for future inputs.

We can apply polynomial regression to multi-dimensional inputs.

MSE loss function has a unique global optimum.



#### Deep Neural Networks

Suppose $\phi(\boldsymbol x)$ has its own set of parameters $\mathbf V$, the model has the overall form:
$$
f(\boldsymbol x; \boldsymbol w, \mathbf V) = \boldsymbol w^\top\phi(\boldsymbol x;\mathbf V).\tag{17}
$$
It is called **nonlinear feature extraction**. We can recursively decompose the feature extractor into a composition of simpler functions. The resulting model becomes a stack of nested functions:
$$
f(\boldsymbol x; \boldsymbol \theta)=f_L(f_{L-1}(\cdots(f_1(\boldsymbol x))\cdots)), \tag{18}
$$
where $f_\ell (\boldsymbol x)$ is the function at layer $\ell$. The final layer is linear and has the form
$$
f_L(\boldsymbol x)=\boldsymbol w^\top f_{1:L-1}(\boldsymbol x). \tag{19}
$$
This is the key idea behind **deep neural networks (DNN)**.

---

### Overfitting and Generalization

A model that perfectly fits the training data, but which is too complex, is said to suffer from **overfitting**.

To detect if a model is overfitting, assume that we can access the true distribution $p^*(\boldsymbol x,\boldsymbol y)$ used to generate the training set. Then we can compute the **population risk**:
$$
\mathcal L(\boldsymbol \theta; p^*) \triangleq \mathbb E_{p^*(\boldsymbol x, \boldsymbol y)}[\ell(\boldsymbol y,f(\boldsymbol x;\boldsymbol \theta))].\tag{20}
$$
The difference between the population risk and the empirical risk $\mathcal L(\boldsymbol \theta; p^*) - \mathcal L(\boldsymbol \theta; \mathcal D_\text{train})$ is called the **generalization gap**. A larger generalization gap is a sign that it is overfitting.

Since we do not known $p^*$ in practice, we can partition the data we do have into two subsets, known as the training set and the **test set**. Then the population risk can be approximated by the **test risk**:
$$
\mathcal L(\boldsymbol \theta; \mathcal D_\text{test}) \triangleq \frac{1}{\vert\mathcal D_\text{test}\vert}\sum_{(\boldsymbol x, \boldsymbol y)\in \mathcal D_\text{test}}\ell(\boldsymbol y,f(\boldsymbol x;\boldsymbol \theta)).\tag{21}
$$
The test error has characteristic U-shaped curve with respect to $D$. Let $D' \triangleq \underset{D}{\text{argmin}}\,\mathcal L(\boldsymbol \theta; \mathcal D_\text{test})$. If $D=D'$, the model is "just right". When $D\ll D'$, the model is **underfitting**. When $D\gg D'$, the model is **overfitting**.

In practice, we need to partition the data into three sets: the training set, the test set and a **validation set**. The latter is used for model selection, and the test set is used only for evaluation.

---

### No Free Lunch Theorem

There is no single best model that works optimally for all kinds of problems. A model that works in one domain may work poorly in another.



## Unsupervised Learning

### Definition and Advantages

#### Definition

The task $T$ is to observe input $\mathcal D=\{\boldsymbol x_n\}$ and fit an **unconditional model** $p(\boldsymbol x)$.



#### Advantages

- It avoids the need to collect labeled datasets for training.
- It avoids the need to learn how to partition the world into arbitrary categories.
- It forces the model to explain the high-dimensional inputs.

---

### Clustering

The goal is to partition the input into regions that contains similar points. We need to consider the tradeoff between model complexity and fit to the data.

---

### Discovering Latent "Factors of Variation"

When dealing with high-dimensional data, it is useful to reduce the dimensionality by projecting it to a lower dimensional subspace.

We can assume that each observed high-dimensional output $\boldsymbol x_n\in\mathbb R^D$ was generated by a set of hidden low-dimensional latent factors $\boldsymbol z_n\in\mathbb R^K$. The model can be represented as $\boldsymbol z_n\to\boldsymbol x_n$, where the arrow represents **causation**.

When we use a linear model,
$$
p(\boldsymbol x_n | \boldsymbol z_n;\boldsymbol \theta)=\mathcal N(\boldsymbol x_n | \mathbf W \boldsymbol z_n + \boldsymbol \mu, \mathbf \Sigma). \tag{22}
$$
When $$\mathbf \Sigma = \sigma^2 \mathbf I$$, the model is called probabilistic **principal components analysis (PCA)**.

---

### Self-Supervised Learning

We can create proxy supervised tasks from unlabeled data. A predictor $\boldsymbol{\hat x}_1=f(\boldsymbol x_2;\boldsymbol \theta)$ is used to generate predicted output, which can be used in downstream supervised tasks.

---

### Evaluating Unsupervised Learning

We can evaluate the model by computing the unconditional NLL of the data:
$$
\mathcal L(\boldsymbol \theta;\mathcal D)=-\frac{1}{|\mathcal D|}\sum_{\boldsymbol x \in \mathcal D}\log p(\boldsymbol x|\boldsymbol \theta). \tag{23}
$$
The treats the problem of unsupervised learning as one of **density estimation**. A good model will not be surprised by actual data samples and assign them with high probability.

An alternative evaluation metric is to use the learned unsupervised representation as features or input to a downstream supervised learning method.



## Reinforcement Learning

The system or **agent** has to learn how to interact with the environment. This can be encoded by means of a **policy** $\boldsymbol a=\pi(\boldsymbol x)$, which specifies which action to take in response to each possible input $\boldsymbol x$.

The system receives an occasional **reward** signal in response to the action that it takes.

It is common to use other information sources such as expert demonstrations, which can be used to discover the underlying structure of the environment.



## Data

### Image Datasets

|      Name       |               Content                | Num. Classes |  Data Representation   |       Dataset Size       |                             Note                             |
| :-------------: | :----------------------------------: | :----------: | :--------------------: | :----------------------: | :----------------------------------------------------------: |
|      MNIST      |          Handwritten digits          |      10      | 28×28 grayscale images | 60k training + 10k test  |        Is called the "drosophila of machine learning"        |
|     EMNIST      | MNIST + upper and lower case letters |      62      |     Same as MNIST      | ~70k training + 12k test |                                                              |
|  Fashion-MNIST  |          Pieces of clothing          |      10      |     Same as MNIST      |      Same as MNIST       |                                                              |
|    CIFAR-10     |           Everyday objects           |      10      |   32×32×3 RGB images   | 50k training + 10k test  |                                                              |
| ImageNet Subset |           Everyday objects           |      1k      |  256×256×3 RGB images  |      ~1.3M training      | The task is to ensure the correct label is within the 5 most probable predictions |

---

### Text Datasets

ML is often applied to text solve a variety of tasks, which is known as **natural language processing (NLP)**.

#### Text Classification

Text classification can be used for **email spam classification**, **sentiment analysis**, etc. A common dataset for evaluating such methods is the **IMDB movie review set**, containing 25k labeled training examples and 25k test examples. Each example has a binary label representing a positive or negative rating.



#### Machine Translation

The task is to learn to map a sentence $\boldsymbol x$ in one language to a "semantically equivalent" sentence $\boldsymbol y$ in another language. The **WMT** contains English-German pairs, and is widely used as a benchmark dataset.



#### Other Seq2Seq Tasks

The task is to learn a mapping from one sequence $\boldsymbol x$ to any other sequence $\boldsymbol y$. It includes tasks such as **document summarization**, **question answering**, etc.



#### Language Modeling

In language modeling, there are no labels. This is a form of unsupervised learning.

---

### Processing Discrete Input Data

#### One-hot Encoding (Dummy Encoding)

$$
\text{one-hot}(x) \triangleq [\mathbb I(x=1),\cdots,\mathbb I(x=K)]. \tag{24}
$$



#### Feature Crosses

$$
\phi(x) \triangleq [1, \mathbb I(x_1=1), \cdots,\mathbb I(x_1 = K),\mathbb I(x_2=1), \cdots,\mathbb I(x_2 = L),\cdots,\mathbb I(x_1=1,x_2=1),\cdots,\mathbb I(x_1=K,x_2=L)]. \tag{25}
$$

---

### Text Data Processing

#### Bag of Words Model

The model is to map each word to a token from some vocabulary.

Pre-processing techniques to reduce the number of tokens:

- Dropping punctuation;
- Converting all words to lower case;
- Dropping common but uninformative words ("and", "the"), which is also called **stop word removal**;
- Replacing words with their base form, which is also called **word stemming**.

We can represent the $n$'th document as a $D$-dimensional vector $\tilde{\boldsymbol{x}}_{n}\in\mathbb R^D$, where $\tilde x_{nv}$ is the number of times that the word $v$ occurs in the document $n$:
$$
\tilde x_{nv}=\sum_{t=1}^T\mathbb I(x_{nt}=v), \tag{26}
$$
where $T$ is the length of the document.

We can store the input data in an $N\times D$ matrix denoted by $\mathbf X$. It is more common to represent the input data as a $D\times N$ **term frequency matrix**, where $\text{TF}_{ij}$ is the frequency of the term $i$ in document $j$.

<center>
    <img src = "https://github.com/takahiroChen/blog-img/blob/main/20220529085814.png?raw=true"
         style = "zoom:40%">
    <br>
    Illustration of the Term Frequency Matrix
</center>



#### TF-IDF

Frequent words may do not carry more semantic information. A common way is to transform the counts by taking logs.

We compute **inverse document frequency** defined as
$$
\text{IDF}_i \triangleq \log\frac{N}{1+\text{DF}_i}, \tag{27}
$$
where $\text{DF}_i$ is the number of documents with term $i$.

We can combine it with TF to compute the **TF-IDF** matrix:
$$
\text{TFIDF}_{ij}=\log(\text{TF}_{ij}+1)\times\text{IDF}_i. \tag{28}
$$


#### Word Embeddings

Word embeddings map each sparse one-hot vector $\boldsymbol x_{nt}\in\{0,1\}^V$ to a lower-dimensional vector $\boldsymbol e_{nt}\in\mathbb R^K$ using $\boldsymbol e_{nt}=\mathbf E\boldsymbol x_{nt}$, where $\mathbf E\in \mathbb R^{K\times V}$ is learned such that semantically similar words are placed close by.

We can represent a variable-length text document as a **bag of word embeddings**:
$$
\bar{\boldsymbol e}_n=\sum_{t=1}^T \boldsymbol e_{nt}=\mathbf E \tilde{\boldsymbol x}_n. \tag{29}
$$
We can use it into a classifier, with the overall model:
$$
p(y=c|\boldsymbol x_n; \boldsymbol \theta)=\mathcal S_c(\mathbf{WE}\tilde{\boldsymbol x}_n). \tag{30}
$$


#### Dealing with Novel Words

The model may encounter a completely novel word. This is known as **out of vocabulary (OOV)** problem. A standard solution is to replace the word with a UNK token, or we can find the substructure of the word to infer its meaning.

---

### Handling Missing Data

Sometimes we may have missing data, in which parts of the input $\boldsymbol x$ or the output $y$ may be unknown, which can be considered as **semi-supervised** learning.

We can model the problem into an $N\times D$ matrix $\mathbf M$, where
$$
M_{nd}=
\begin{cases}
1 & \text{if feature } d \text{ in example } n \text{ is missing}\\
0 & \text{otherwise}.
\end{cases} \tag{31}
$$
Let $\mathbf X_v$ be the visible parts of the input feature matrix, and $\mathbf X_h$ be the missing parts. Let $\mathbf Y$ be the output label matrix. If $p(\mathbf M|\mathbf X_v,\mathbf X_h, \mathbf Y)=p(\mathbf M)$, the data is **missing completely at random (MCAR)**. If $p(\mathbf M|\mathbf X_v,\mathbf X_h, \mathbf Y)=p(\mathbf M | \mathbf X_v, \mathbf Y)$, the data is **missing at random (MAR)**. Otherwise, the data is **not missing at random (NMAR)**. In NMAR case, we need to model the **missing data mechanism**, as they may contain underlying information.

A common heuristic is called **mean value imputation**, where missing values are replaced by their empirical mean.
