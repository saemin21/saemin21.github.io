---
title: Deep Learning 3.Probability and Information Theory
date: 2024-10-22 16:44:20 +0900
categories:
  - study
  - deep learning (AIGS538)
tags:
  - deep_learning
  - lecture
  - study
  - AIGS538
description: 
toc: true
comments: false
media_subpath: /assets/img/
cdn: 
image: 
math: true
pin: false
mermaid: false
---

영어로 쓰면 시간이 오래 걸려 이제부터 한글로..

# Probability and Information Theory

![Desktop View](deep_learning/org_1.png)_Organization of the book_


Probability theory
: allows us to make **uncertain statements** and to reason in the presence of uncertainty.

Information theory
: enables us to **quantify** the amount of uncertainty in a probability distribution.

## 3.1 Why Probability?

Probability는 deep neural networks를 만드는 데에 필수적이다.

Machine learning은 **uncertain** quantities 혹은 **stochastic** (non-deterministic) quantities를 다루기 때문에 이 uncertainty를 정량화하는 방법으로 probability를 사용하는 것이다.

There are three possible sources of uncertainty:
1. Inherent stochasticity in the system being modeled.
2. Incomplete observability.
3. Incomplete modeling.

많은 경우에 certain하지만 복잡한 rule을 사용하는 것보다 uncertain하지만 간단한 rule를 사용하는 것이 실용적이다.
(e.g., "Most birds fly" vs "Birds fly, except for very young birds that have ... sick or injured birds ...")

Probability에는 두 가지 종류가 있는데,

Frequentist probability
: A rate at which events occur.

Bayesian probability
: Qualitative levels of certainty.

어차피 same formal rules가 적용된다. (자세한 것은 (*Ramsey, 1926*))

## 3.2 Random Variables

Measure theory에서 좀 더 deep하게 다룰 수 있지만, 일단은 값들을 랜덤하게 가질 수 있는 variable.
RV 하나로는 어떤 state들이 가능한지에 대한 description이기 때문에 probability distribution과 페어를 이루어야 각 state들의 likelihood를 specify할 수 있음.

lowercase script letter로 씀.

Discrete or continuous.

## 3.3 Probability Distributions

Probability distribution은 how likely a random variable (or set)이 각 possible states를 취하는 지에 대한 description.

Variable이 discrete하냐, continuous 하냐에 따라 다름.

### 3.3.1 Discrete Variables and Probability Mass Functions

Probability Mass Function (**PMF**)
: A probability distribution over discrete variables may be described using a probability mass function.

E.g., Uniform distribution: $P(\mathbf x = x_i) = {1\over k}$.

Properties:
- P의 domain은 $\mathbf x$의 모든 possible states의 set이어야 한다.
- $\forall x \in \mathbf x, 0 \le P(x) \le 1$. 즉 모든 확률은 0과 1사이어야 함.
- $\sum_{x\in \mathbf x} P(x) = 1$. 모든 possible states의 확률 sum은 1. (normalized)

### 3.3.2 Continuous Variables and Probability Density Functions

variable이 continuous하다면 조근 다르게 density를 통해 표현해야 한다.

Probability Density Function (**PDF**)
: A probability distribution over continuous variables may be described using a probability density function.

E.g., Uniform distribution: $u(x;a,b) = {1\over b-a}$.
(;는 "parameterized by"를 의미함.)

Properties:
- PMF와 마찬가지로 p의 domain은 모든 possible states의 set.
- $\forall x \in \mathbf x, p(x) \ge 0$.
	- (density이기 때문에 1 이하라는 constraint는 없어도 됨.)
- $\int p(x) dx = 1$
	- Given interval $[a, b]$, $\int_{[a,b]} p(x) dx = 1$

## 3.4 Marginal Probability

Marginal probability는 RV들의 **subset**에 대한 probability distribution을 일컫는다.
밑의 $\mathbf y$가 일종의 subset(?)

The sum rule
: $$
\begin{align}
\forall x \in \mathbf x, P(\mathbf x=x)&=\sum_y P(\mathbf x = x, \mathbf y = y) \\
p(x) &= \int p(x,y)dx
\end {align}
$$

각각 discrete, continuous variable에 관한 sum rule.

## 3.5 Conditional Probability

주어진 다른 event가 있을 때의 probability를 말한다.

$$P(\mathbf y = y \mid \mathbf x = x) = {P(\mathbf y = y, \mathbf x = x) \over P(\mathbf x = x)}$$

## 3.6 The Chain Rule of Conditional probabilities

Any joint probability는 chain rule을 통해 아래처럼 conditional distributions로 decompose할 수 있다.

$$
P(\mathbf x^{(1)}, \dots , \mathbf x ^{(n)})
= P(\mathbf x^{(1)} \Pi^n_{i=2} P(\mathbf x^{(i)} \mid \mathbf x^{(1)}, \dots, \mathbf x^{(i-1)}).
$$

더 쉬운 아래 예시.
$$
P(a,b,c) = P(a \mid b,c) P(b,c) = P(a \mid b,c) P(b \mid c )P(c)
$$

## 3.7 Independence ad Conditional Independence

## 3.8 Expectation, Variance and Covariance

## 3.9 Common Probability Distributions

### 3.9.1 Bernoulli Distribution

### 3.9.2 Multinoulli Distribution

### 3.9.3 Gaussian Distribution

### 3.9.4 Exponential and Laplace Distributions

### 3.9.5 The Dirac Distribution and Empirical Distribution

### 3.9.6 Mixtures of Distributions

## 3.10 useful Properties of Common Functions

## 3.11 Bayes' Rule

## 3.12 Technical Details of Continuous Variables

## 3.13 Information Theory

## 3.14 Structured Probabilistic Models