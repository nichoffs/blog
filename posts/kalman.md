---
title: "The Kalman Filter"
date: 2025-03-09T20:49:49+00:00
draft: false
---

# Finally...

We've gotten to the Kalman Filter. Up to this point, I've covered markov models, hidden markov models and the problems that can be solved with them (filtering, smoothing, decoding, etc), Bayes filters, Linear Algebra, and the strategy for optimally combining estimators linearly. In this post, I will bring everything together with the Kalman Filter.

To put it concisely, the Kalman Filter is the optimal solution to the Bayes Filter for systems with linear dynamics and Gaussian Noise. What does this mean? Well let me review:

## Markov Models, Hidden Markov Models, and the new Markov Decision Process (MDP)

An MDP describes a "stochastic dynamic system" -- a process where our state transitions from one to another depending on some control input, yet there is noise associated with the state transition. We are unsure of whether or not the stated control input will translate to the update in state that we are expecting. If our state transition dynamics are described by a function $f$, the markov decision process can be represented mathematically as:

$$
x_{k+1}=f(x_k, u_k)+\epsilon_k
$$

Since the transition is stochastic, not deterministic, the state transition can alternatively be represented as a probability distribution:

$$
x_{k+1}=P(x_{k+1}|x_k,u_k)
$$

This situation is similar to the traditional Markov Model except that, for each control input, we have a separate transition matrix.

Lastly, since this situation describes the system modeled by the Kalman Filter, a linear stochastic system is formulated as follows:

$$
x_{k+1}=Ax_k+Bu_k+\epsilon_k
$$

Consider one more thing -- our observations in an HMM do not necessarily directly represent our state. It may be the case that our state is position and our GPS measures this position with some noice. However, it may be that the relation is less direct. What is the analogy for a more linear stochastic system? ** MORE ANALOGY ** 

## Incorporating Gaussian Observations of a State Linearly

Imagine our sensor relates to the d-dimensional state through a linear function but it has some zero-mean Gaussian noise associated with it:

$$
\mathbb{R}^p \ni Y=CX+v
$$

Notice the dimensionality of $Y$ is $\mathbb{R}^p$ but our state $X$ has dimension $\mathbb{R}^d$. Therefore, our $C$ matrix must have shape $p \times d$. Additionally, we have $v \sim N(0,Q)$.

It's similar to how, in an HMM, our emission matrix $M$ may not be square. We could have more observations than state, or less.

New Estimate:

$$
\begin{align*}
\hat{X} = K'\hat{X}' + KY = K'\hat{X}' + K(C\hat{X}'+v)\\
E[\hat{X}] = E[K'\hat{X}' + KY] = E[K'\hat{X}' + K(C\hat{X}'+v)] = K'E[\hat{X}'] + KE[C\hat{X}'+v] = K'E[\hat{X}'] + KCE[\hat{X}'] + KE[v]\\
K'\hat{X}' + KC\hat{X}' = \hat{X}'\\
K' + KC = I\\
K' = I - KC
\end{align*}
$$

Substituting:

$$
\begin{align*}
\hat{X} = (I-KC)\hat{X}' + KY\\
= \hat{X}' - KC\hat{X}' + KY \\
= \hat{X}' + K(Y-C\hat{X}')
\end{align*}
$$

Covariance using independence assumption:

$$
\begin{align*}
\Sigma_{\hat X}=\text{Cov}((I-KC)\hat X' + KY)\\
= (I-KC)\Sigma_{\hat X'} (I-KC)^T + KQK^T
\end{align*}
$$

Now how can we minimize the trace of the covariance of the updated estimator with respect to $K$? My matrix calculus is not solid enough for this, so I'll have to take their word:

$$
0 = \frac{\partial}{\partial K} \text{tr}({\Sigma}_{\hat X})
$$

$$
0 = -2(I - K C) {\Sigma}_{\hat X'} C^T + 2K Q
$$

$$
\Rightarrow {\Sigma}_{\hat X'} C^T = K (C \hat{\Sigma}_{X'} C^T + Q)
$$

$$
\Rightarrow K = {\Sigma}_{\hat X'} C^T (C {\Sigma}_{\hat X'} C^T + Q)^{-1}.
$$

Now notice what happens if we set $C=I$ (the observation directly estimates state):


$$
K = {\Sigma}_{\hat X'} C^T (C {\Sigma}_{\hat X'} C^T + Q)^{-1}={\Sigma}_{\hat X'} ({\Sigma}_{\hat X'} + Q)^{-1}
$$

Substituting back into our previous formula:

$$
\hat X = K'\hat X'+({\Sigma}_{\hat X'} ({\Sigma}_{\hat X'} + Q)^{-1})Y=\hat X + ({\Sigma}_{\hat X'} ({\Sigma}_{\hat X'} + Q)^{-1})(Y-C\hat X')
$$

$$
\hat X = K'\hat X'+({\Sigma}_{\hat X'} ({\Sigma}_{\hat X'} + Q)^{-1})Y
$$

## Kalman Filter

Now that we know how to update an estimator using an observation which is a linear function of the hidden state, I can now introduce the Kalman Filter.

The system can be described as follows:

$$
x_{k+1}=Ax_{k}+Bu_k+\epsilon_k \\ 
y_k=Cx_k+v_k
$$

with $\epsilon_k \sim N(0,R)$ and $v_k \sim N(0, Q)$.

The first equation describes how the state evolves over time. As you can see, the future state without incoorporating observation is a linear function of the input. Since everything is linear, all states will be gaussians. Therefore, they can be described completely using the mean and covariance. As a result, we only need to figure out what the $\mu$ and $\Sigma$ are at each step.

The Kalman Filter proceeds in two distinct steps, similar to the Bayes Filter. First, the state is propoagated independently of any observations using our model of the system's dynamics. Next, when we get an observation, we update the state estimate. There's some notation I want to introduce to make this a bit clearer:

$$
\hat x_{k|k} \sim P(x_k|y_1,\dots,y_k) \sim N(\mu_{k|k}, \Sigma_{k|k})
$$

$$
\hat x_{k+1|k} \sim P(x_{k+1}|y_1,\dots,y_k) \sim N(\mu_{k+1|k}, \Sigma_{k|k})
$$

Therefore, given $x_{k|k}$ and $y_{k+1}$ we want to get $x_{k+1|k+1}$. This requires us to first propagate $x_{k|k}$ to $x_{k+1|k}$, and then factor observation $y_{k+1}$.

### Propagation

First, we want to find $x_{k+1|k}$, or $P(x_{k+1}|y_1)$:

$$
\hat x_{k+1|k} = A \hat x_{k|k}+Bu_k + \epsilon_k
$$

$$
\mu_{k+1|k}=E[\hat x_{k+1|k}]=E[A \hat x_{k|k}+Bu_k+\epsilon_k]=A \mu_{k|k}+Bu_k \\
\Sigma_{k+1|k}=\text{Cov}(\hat x_{k+1|k})=\text{Cov}(A \hat x_{k|k}+Bu_k+\epsilon_k)=A\Sigma_{k|k}A^T + R
$$

### Observation

Now we have $x_{k+1|k}$ but need to update with our most recent observation.

$$
y_{k+1}=Cx_{k+1}+v_{k+1}
$$

So recall our formula from before:

$$
x_{k+1|k+1}=x_{k+1|k}+K_{k+1}(y_{k+1}-Cx_{k+1|k}) \\ 
$$

Now we need to propagate the means:

$$
\mu_{k+1|k+1}=\mu_{k+1|k}+K_{k+1}(y_{k+1}-C\mu_{k+1|k}) \\ 
\Sigma_{k+1|k+1}=(I-K_{k+1}C)\Sigma_{k+1|k}(I-K_{k+1}C)^T+K_{k+1}QK_{k+1}^T
$$

Or we can used the condensed form for $\Sigma$

$$
\Sigma_{k+1|k+1}=(\Sigma_{k+1|k}^{-1}+C^TQ^{-1}C)^{-1}
$$

Notice that we really have two different formulations of the Kalman Filter update step:

$$
\text{New Estimate} = K'X+KY=(I-KC)X+KY
$$

and

$$
\text{New Estimate} = X+K(Y-CX)
$$

I generally favor the clarity of the second. It shows that we have some noisy estimate from our dynamics model $X$ and some noisy estimate from our sensor model $Y$, and we essentially want to update the estimate with this noisy observation. The way we do that is by taking the difference of our observation from what our dynamics model would have expected the observation to be (basically the new information added by our observation) and then scaling it by some factor related to both the covariance of the estimate and the observation. If we have a noisier observation, $K$ will be smaller.

By using the form $X+K(Y-CX)$ or, equivalently, $(I-KC)X+KY$, we enforce that our estimated value is unbiased. Then, by minimizing the trace of the covariance of the new estimator with respect to $K$, we find the optimal formulation for combining the estimate and observation.

I hope this is relatively intuitive by this point. Although the matrix calculus is a bit over my head, I think previous examples give sufficient understanding for why the Kalman Filter is the optimal linear estimator for a system with Gaussian noise.

    