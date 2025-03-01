---
title: "Expectation, Covariance, and Optimal Estimators"
date: 2025-02-25T01:46:26-05:00
draft: false
---

# Expected Value and Variance

$X$ can take a range of values. The distribution of outcomes is dependent on the underlying pdf $f(x)$. It has a mean value, also known as the expected value. The mean and expected value are denoted by $\mu_x$ and $E[X]$, respectively.

We can calculate the expected value by doing a weighted sum of the outcome of a random value and its corresponding probability:

$$
E[X]=\int_{-\infin}^{\infin}xf(x)dx
$$

Now we also want a measure of the spread -- how much the outcomes typically deviate from the expected value.

$$
\sigma_x^2=\int_{-\infin}^{\infin}(x-E[x])^2f(x)dx
$$

$\sigma$ is the standard deviation and $\sigma^2$ is the variance.

# Covariance for $X \in \mathbb{R}^d$

Now what if we have a multivariate random variable? Say, for instance, the position of a robot on a 2D Grid: $X = \begin{bmatrix} x \\ y \end{bmatrix}$.

The expected value for the vector can be found simply by taking the expected value element-wise:

$$
E[X] = \begin{bmatrix} \int_{-\infin}^{\infin}(x-E[x])^2f(x)dx \\ \int_{-\infin}^{\infin}(y-E[y])^2f(y)dy \end{bmatrix} 
$$

Variance is actually trickier. The features are related in some way (that's why they're in a single vector) so taking the element-wise will leave valuable information out. We can categorize the relation between the two features into two groups:

- When $x$ is big, $y$ is big. When $x$ is small, $y$ is small.
- When $x$ is big, $y$ is big. When $x$ is small, $y$ is big.

But what do we mean by big or small? Well, the difference between the random variable and its expected value is a good measure! It's small if $X-E[X] < 0$ and big if $X-E[X] > 0$.

Note that it's not possible to have this configuration:
- When $x$ is big, $y$ is big.
- When $x$ is small, $y$ is big.

If this were the case, it'd mean that $y$ is always big! This isn't possible because then the mean would be bigger and there would be more small $y$ values.

The relation between $x$ and $y$ is really what defines the covariance. It tells us a lot about the "shape" of the distribution. But how do we calculate it? Well, we only really care about whether or not the variables change together or not. It doesn't matter if they're both small or both big -- this is one group. We want a positive covariance if they're directly proportional and a negative if they're inversely proportional. Consider the formula for the expected value and it should feel quite intuitive:

$$
\text{Cov}(X,Y)=E[(X-E[X])(Y-E[Y])]
$$

If both values are small or big (negative * negative = positive), the value inside the expected value will be positive. If they're different, it will be negative. The expected value gives us a notion of the average result.

A nice property of covariance is that taking the covariance of a random variable with itself is simply the variance.

$$
\text{Cov}(X,X)=E[(X-E[X])(X-E[X])]\\
\text{Cov}(X,X)=E[(X-E[X])^2]=\int_{-\infin}^{\infin}(X-E[X])^2f(x)dx\\
$$

So for our multidimensional random variable, the **covariance matrix** would look like this:

$$\Sigma_X=\begin{bmatrix} \text{Cov}(x,x) & \text{Cov}(x,y) \\ \text{Cov}(y,x) & \text{Cov}(y,y) \end{bmatrix}=\begin{bmatrix} \text{Var}(x) & \text{Cov}(x,y) \\ \text{Cov}(y,x) & \text{Var}(y) \end{bmatrix}$$

And for a random vector with D dimensions:

$$
\Sigma_X =
\begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_d) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_d) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_d, X_1) & \text{Cov}(X_d, X_2) & \cdots & \text{Var}(X_d)
\end{bmatrix}
$$

This can actually be calculated using a matrix product as follows:

$$
\Sigma_X =
\begin{bmatrix}
E[(X_1 - E[X_1]) (X_1 - E[X_1])] & E[(X_1 - E[X_1]) (X_2 - E[X_2])] & \cdots & E[(X_1 - E[X_1]) (X_d - E[X_d])] \\
E[(X_2 - E[X_2]) (X_1 - E[X_1])] & E[(X_2 - E[X_2]) (X_2 - E[X_2])] & \cdots & E[(X_2 - E[X_2]) (X_d - E[X_d])] \\
\vdots & \vdots & \ddots & \vdots \\
E[(X_d - E[X_d]) (X_1 - E[X_1])] & E[(X_d - E[X_d]) (X_2 - E[X_2])] & \cdots & E[(X_d - E[X_d]) (X_d - E[X_d])]
\end{bmatrix}
$$

$$
X_c =
\begin{bmatrix}
X_1 - E[X_1] \\
X_2 - E[X_2] \\
\vdots \\
X_d - E[X_d]
\end{bmatrix}
$$


$$
X_c^T =
\begin{bmatrix}
X_1 - E[X_1] &
X_2 - E[X_2] &
\cdots &
X_d - E[X_d]
\end{bmatrix}
$$

$$
\Sigma_X = E \left[X_c X_c^T \right]
$$

Feel free to verify this product yourself.

# Properties of Expectation and Covariance

## Linearity

We want to show:

$$
E[aX+bY]=aE[X]+bE[y]
$$

First, note that, using marginalization: $\int_{-\infin}^{\infin}f(x,y)dy=f(x)$. With that, we can proceed with the derivation:

$$
E[aX+bY]=\int_{-\infin}^{\infin}(aX+bY)f(x,y)dxdy=\int_{-\infin}^{\infin}(aX)f(x,y)dxdy+\int_{-\infin}^{\infin}(bY)f(x,y)dxdy\\
=a\int_{-\infin}^{\infin}(X)f(x)dy+b\int_{-\infin}^{\infin}(Y)f(x)dy=aE[X]+bE[y]
$$

## $\text{Cov}(X+Y)= \text{Cov}(X) + \text{Cov}(X,Y) + \text{Cov}(Y,X) + \text{Cov}(Y)$

$$
\begin{align*}
\text{Cov}(X+Y)=E[(X+Y-E[X+Y])(X+Y-E[X+Y])^T]=\\E[(X+Y-E[X]-E[Y])(X+Y-E[X]-E[Y])^T]\\
=E[(X-E[X])(X-E[X])^T+(X-E[X])(Y-E[Y])^T\\+(Y-E[Y])(X-E[X])^T+(Y-E[Y])(Y-E[Y])^T]\\
=E[(X-E[X])(X-E[X])^T]+E[(X-E[X])(Y-E[Y])^T]\\+E[(Y-E[Y])(X-E[X])^T]+E[(Y-E[Y])(Y-E[Y])^T]\\
=\text{Var}(X)+\text{Cov}(X,Y)+\text{Cov}(Y,X)+\text{Var}(Y)\\
=\text{Var}(X)+2\text{Cov}(X,Y)+\text{Var}(Y)\\
\end{align*}
$$

## $\text{Cov}(AX)$

$$
\begin{align*}
\text{Cov}(AX)=E[(AX-E[AX])(AX-E[AX])^T]=E[(AX-AE[X])(AX-AE[X])^T]\\
=E[A(X-E[X])(X-E[X])^TA^T]\\
=AE[(X-E[X])(X-E[X])^T]A^T=A\text{Cov(X)}A^T
\end{align*}
$$

Note that, when I say $\text{Cov}(X)$, this is the covariance of each feature of the random variables with respect to the other features. It's the same case as I explored earlier with $X=\begin{bmatrix} x \\ y \end{bmatrix}$.

# Estimation

Imagine $X \in \mathbb{R}^d$ is a hidden state which we can't observe. Since it's unobservable, we want an estimator $\hat X$. The estimator may be noisy (ideally not), but it definitely shouldn't have bias. An unbiased estimator refers to an estimator which, in the long run, averages out to the value of the hidden state. In other words, despite the noise, the estimator is centered around the hidden states value. 

To put it mathematically, we want $E[\hat X]=X$.

The error of our estimator is $\tilde X=\hat X - X$. The expected value of our estimator should be zero, because the bias is 0 and, on the average, the noise is displaced from the mean evenly on either side:

$$
E[\tilde X]=E[\hat X - X]=E[\hat X]-E[X]=X-X=0.
$$

The covariance of the estimator is $\Sigma_{\hat X}$.

## Combining Estimators

Oftentimes we have two separate estimators of a single hidden state: $\hat X_1, \hat X_2$. How can we combine them to achieve an estimator which is better than either estimator on its own? Well, if we're looking for an optimal combination, we need some measure of optimality to optimize for. The trace of a matrix is the sum of its diagonals. In a covariance matrix, the diagonals are the variances for each of the features of the random variable. Since we really care more about the spread of the random variable, and not really the covariance between features, this seems like a good measurement for the overall noise level.

With that measure of optimality, we can formalize our task:

$$
\hat X = f(\hat X_1, \hat X_2) \text{ s.t. } \text{tr}(\Sigma_{\hat X}) \text{ is minimized and } E[\hat X]=X
$$

So, we assure the estimator is unbiased and then minimze the trace of the covariance.

Consider the example of a linear combination of two 1D Gaussians:

$$
\hat X=f(\hat X_1, \hat X_2)=k_1 \hat X_1 + k_2 \hat X_2
$$

We want it to be unbiased:

$$
\begin{align*}
E[\hat X]=E[k_1 \hat X_1 + k_2 \hat X_2]=k_1E[\hat X_1]+k_2E[\hat X_2]=X\\
k_1X+k_2X=X\\
=k_1+k_2=1\\
k_2=1-k_1
\end{align*}
$$

We assume the two estimators are independent so the covariance is $0$. For independent RV, the following identity applies: $\text{Var}(aX+bY)=a^2\text{Var}(X)^2+b^2\text{Var}(Y)^2$.

$$
\begin{align*}
\text{Var}(\hat X)=\text{Var}(k_1 \hat X_1 + k_2 \hat X_2)=k_1^2 \sigma_1^2+k_2^2\sigma_2^2\\
=k_1^2 \sigma_1^2+(1-k_1)^2\sigma_2^2\\
=k_1^2 \sigma_1^2+(1-2k_1+k_1^2)\sigma_2^2\\
=k_1^2 \sigma_1^2+\sigma_2^2-2k_1\sigma_2^2+k_1^2\sigma_2^2
\end{align*}
$$

Now we want to minimize the variance w.r.t $k_1$:

$$
\begin{align*}
\frac{\partial}{\partial k_1}(k_1^2 \sigma_1^2+\sigma_2^2-2k_1\sigma_2^2+k_1^2\sigma_2^2)=0\\
2k_1\sigma_1^2-2\sigma_2^2+2k_1\sigma_2^2=0\\
k_1\sigma_1^2-\sigma_2^2+k_1\sigma_2^2=0\\
k_1=\frac{\sigma_2^2}{\sigma_1^2+\sigma_2^2}
\end{align*}
$$

Solving for $k_2$:

$$
\begin{align*}
k_2=1-\frac{\sigma_2^2}{\sigma_1^2+\sigma_2^2}\\
=\frac{\sigma_1^2+\sigma_2^2}{\sigma_1^2+\sigma_2^2}-\frac{\sigma_2^2}{\sigma_1^2+\sigma_2^2}\\
=\frac{\sigma_1^2}{\sigma_1^2+\sigma_2^2}
\end{align*}
$$

Replacing $k_1,k_2$ with the critical values we calculated:

$$
\hat X = \frac{\sigma_2^2}{\sigma_1^2+\sigma_2^2} \hat X_1 + \frac{\sigma_1^2}{\sigma_1^2+\sigma_2^2} \hat X_2
$$

This should intuitively make sense. If \hat X_1 has a lot of noise ($\sigma_1$ is big), $\hat X_2$ will contribute more to the estimator.

## Combining Estimators for multidimensional RV

Now we have $X,\hat X_1, \hat X_2, \hat X \in \mathbb{R}^d$ and $K_1,K_2 \in \mathbb{R}^{d \times d}$:

$$
\hat X = K_1 \hat X_1 + K_2 \hat X_2
$$

First, we must ensure the estimator is unbiased:

$$
\begin{align*}
E[\hat X]=E[K_1 \hat X_1 + K_2 \hat X_2]=K_1E[\hat X_1]+K_2E[\hat X_2]=K_1X+K_2X=X\\
K_1+K_2=I\\
K_2=I-K_1
\end{align*}
$$

This condition will ensure the new estimator is unbiased. Onto the minimization of covariance:

$$
\begin{align*}
\Sigma_{\hat X}=\text{Cov}(\hat X)=\text{Cov}(K_1\hat X_1+K_2 \hat X_2)=\text{Cov}(K_1 \hat X_1)+\text{Cov}(K_2 \hat X_2)\\
=K_1 \Sigma_{\hat X_1}K_1^T + K_2 \Sigma_{\hat X_2}K_2^T\\
=K_1 \Sigma_{\hat X_1}K_1^T + (I-K_1) \Sigma_{\hat X_2}(I-K_1)^T\\
=K_1 \Sigma_{\hat X_1}K_1^T + K_1 \Sigma_{\hat X_2}K_1^T-K_1\Sigma_{\hat X_2}-\Sigma_{\hat X_2}K_1^T+\Sigma_{\hat X_2}
\end{align*}
$$

$$
\begin{align*}
\text{Tr}(\Sigma_{\hat X})=\text{Tr}(K_1 \Sigma_{\hat X_1}K_1^T) + \text{Tr}(K_1 \Sigma_{\hat X_2}K_1^T)-\text{Tr}(K_1\Sigma_{\hat X_2})-\text{Tr}(\Sigma_{\hat X_2}K_1^T)+\text{Tr}(\Sigma_{\hat X_2})\\
=\text{Tr}(K_1 \Sigma_{\hat X_1}K_1^T) + \text{Tr}(K_1 \Sigma_{\hat X_2}K_1^T)-\text{Tr}(K_1\Sigma_{\hat X_2})-\text{Tr}(K_1\Sigma_{\hat X_2})+\text{Tr}(\Sigma_{\hat X_2})\\
=\text{Tr}(K_1 \Sigma_{\hat X_1}K_1^T) + \text{Tr}(K_1 \Sigma_{\hat X_2}K_1^T)-2\text{Tr}(K_1\Sigma_{\hat X_2})+\text{Tr}(\Sigma_{\hat X_2})=
\end{align*}
$$

Now taking the partial derivative w.r.t $K_1$:

$$
\begin{align*}
\frac{\partial}{\partial K_1}(\text{Tr}(K_1 \Sigma_{\hat X_1}K_1^T) + \text{Tr}(K_1 \Sigma_{\hat X_2}K_1^T)-2\text{Tr}(K_1\Sigma_{\hat X_2})+\text{Tr}(\Sigma_{\hat X_2}))=0\\
\frac{\partial}{\partial K_1}(\text{Tr}(K_1 \Sigma_{\hat X_1}K_1^T) + \text{Tr}(K_1 \Sigma_{\hat X_2}K_1^T)-2\text{Tr}(K_1\Sigma_{\hat X_2}))=0\\
2K_1\Sigma_{\hat X_1} + 2K_1\Sigma_{\hat X_2}+\frac{\partial}{\partial K_1}(-2\text{Tr}(K_1\Sigma_{\hat X_2}))=0\\
2K_1\Sigma_{\hat X_1} + 2K_1\Sigma_{\hat X_2}-2\Sigma_{\hat X_2}=0\\
K_1\Sigma_{\hat X_1} + K_1\Sigma_{\hat X_2}-\Sigma_{\hat X_2}=0\\
K_1(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})=\Sigma_{\hat X_2}\\
K_1=\Sigma_{\hat X_2}(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1}\\
\end{align*}
$$

Solving for $K_2$:

$$
\begin{align*}
K_2=I-K_1=I-\Sigma_{\hat X_2}(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1}\\
K_2=(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1}-\Sigma_{\hat X_2}(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1}\\
K_2=(\Sigma_{\hat X_1} + \Sigma_{\hat X_2}-\Sigma_{\hat X_2})(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1}\\
K_2=(\Sigma_{\hat X_1})(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1}\\
\end{align*}
$$

Rewriting our original expression for the estimator:

$$
\hat X = \Sigma_{\hat X_2}(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1} \hat X_1 + (\Sigma_{\hat X_1})(\Sigma_{\hat X_1} + \Sigma_{\hat X_2})^{-1} \hat X_2
$$

This looks really similar to the 1-D case and still aligns with intuition. If $\hat X_1$ has high covariance, $\hat X_2$ will be large and dominate in the new estimator. 

This choice of $K_1$ and $K_2$ is actually optimal for all estimators given that the estimators, as the function we're optimizing is convex. I don't know that much about this, so I won't go into great detail.

The method of optimizing with respect to the trace of the covariance is foundational to the derivation for the Kalman Filter, so understanding this prerequisite is critical.
    