---
title: "The Bayes Filter in Different Contexts"
date: 2025-02-16T17:58:52-05:00
draft: false
excerpt: "Bayes Filters using HMMs and a more general motion model"
---

# Bayes Filter

The Bayes Filter is a general algorithm for state estimation. A Kalman Filter, for example, is a specific implementation of the Bayes Filter and may be the most important algorithm in robotics. Due to the generality of the Bayes Filter, it was difficult for me to grasp the different applications. I've found that studying the implementation in different contexts helped me quite a bit. In this post, I'll cover two different scenarios where a Bayes Filter is useful: state estimation with a Hidden Markov Model (beginning with a regular Markov Model) and with an "Action Model".

## Markov Model - Transitioning between states without observation

WARNING: All of this is a review of my previous post on Markov Chains. For a more detailed breakdown, go there.

Our system can be in one of two states. The likelihood of transitioning to one state from another is summarized in a transition matrix:

$$
T =
\begin{bmatrix}
0.4 & 0.6 \\
0.1 & 0.9
\end{bmatrix}=\begin{bmatrix}
P(X_{t+1}=0|X_t=x=0) & P(X_{t+1}=1|X_t=x=0) \\
P(X_{t+1}=0|X_t=x=1) & P(X_{t+1}=1|X_t=x=1)
\end{bmatrix}
$$

If we're certain about the starting probability distribution (we know what state we're in), $P(X_1)=\pi_1=[P(X_1=0), P(X_1=1)]=[1, 0]$. We need to use the law of total probability to compute the probability of the next state $P(X_2)$ or $\pi_2$:

$$
P(X_2)=[P(X_2=0|X_1=0)P(X_1=0)+P(X_2=0|X_1=1)P(X_1=1),P(X_2=1|X_1=0)P(X_1=0)+P(X_2=1|X_1=1)P(X_1=1)]
$$

Equivalently:

$$
\pi_2=[T_{00}\pi_1^{(0)}+T_{10}\pi_1^{(1)}, T_{01}\pi_1^{(0)}+T_{11}\pi_1^{(1)}]
$$

This means $\pi_2^{(i)}$ is just the dot product of $\pi_1$ and the $i$-th column of $T$. Therefore, in general terms, we can compute $\pi_t$ by taking the dot product of $T'$  and $\pi_{t-1}$ where $T'$ is the transpose:

$$
\pi_t=T'\pi_{t-1}
$$

As $t \rightarrow \infin$, the distribution stops changing which gives us the steady-state distribution of the system. We can approximate the steady-state distribution by running the transition propagation algorithm many times, or exploit the fact that the distribution won't change as follows:

$$
\pi_\infin=T'\pi_{\infin}
$$

Therefore, the steady state distribution is just the eigenvector of $T'$ for an eigenvalue $1$.

Note that this is just a description of how the state would evolve without any observations that might help us refine our judgment. I haven't yet introduced the "hidden" aspect which lets us exploit observations -- so far it's just a Markov Model. At this point, the state at any timestep $t$ can be fully described by the starting state (which may be initialized using the steady-state distribution) and the transition matrix:

$$P(X_t=x|X_0,T)$$

By computing this value for every state $x$, we get the probability distribution of the state at time $t$. This is what we want.

The markov model provides us a very clean method of representing the state of the system and how it evolves. It's all packaged up nicely. The action model is a bit less elegant, although it is more generally applicable.

Under the markov assumption, $X_t$ depends only on $X_{t-1}$. However, in the above conditional probability, $X_t$ seems to depend on $X_0$ and $T$. Because we don't know what $X_{t-1}$ will be, we must marginalize over all possible values of $X_{t-1}$ to express $P(X_t|X_0,T)$ in a form which can be computed recursively and shows the Markov assumption more clearly. For the rest of this section, I won't show the explicit dependence on $T$.

$$
P(X_t=x|X_0)=\sum_{x'}P(X_t=x|X_{t-1}=x')P(X_{t-1}=x'|X_0)
$$

Oftentimes, the prior $X_0$ is also assumed implicitly:

$$
P(X_t=x)=\sum_{x'}P(X_t=x|X_{t-1}=x')P(X_{t-1}=x')
$$

I'll go with this notation because it's pretty clear that the recursion begins with $P(X_0)$. Finally, we can express the conditional in terms of the transition matrix:

$$
P(X_t=x)=\sum_{x'}T_{x'x}P(X_{t-1}=x')
$$

## Action Model -- Transitioning between states without observation

In the action model, our state is the position of a robot in a $2 \times 2$ grid. We are given a list of actions $u_{1:t}$ over the timesteps along with probabilities related to how an action affects the state. This is known as the **motion model**, similar to $T$ in the hidden markov model. A robot can either move up, down, left, or right. It succeeds with probability $.9$ and fails with probability $.1$. If the action would move the robot off the grid, it stays in place with absolute certainty ($1$).

I think this approach lends itself more to a code implementation, although I won't get into those details in this post.

If our set of actions is $u_{1:t}$, then our probability of interest is the state at time $t$:

$$
P(X_t=x|X_0, u_{1:t})
$$

Once again, we can assume the prior implicitly: $P(X_t=x|u_{1:t})$. Similar to the HMM, the dependence on the entire sequence of actions reflects the fact that there are many possible previous states $X_{t-1}$ given our model. If we knew the previous state, we could describe the current state just in terms of the previous state and the current action $P(X_t=x|X_{t-1},u_t)$. 

If we didn't know the previous state, we would marginalize over possible previous states:

$$
P(X_t=x|u_t)=\sum_{x'}P(X_t=x|X_{t-1}=x', u_t)P(X_{t-1}=x'|u_{t-1})
$$

To align more with the HMM notation, we could say that $P(X_t)$ implicitly considers the current action and the same for the previous state, therefore expressing it as such:

$$
P(X_t=x)=\sum_{x'}P(X_t=x|X_{t-1}=x', u_t)P(X_{t-1}=x')
$$

However, it's convention to show that the distribution of the current state depends on the history of all previous states, so I will depart from the HMM notation from here on out. The idea is the exact same.


$$
P(X_t=x|u_{1:t})=\sum_{x'}P(X_t=x|X_{t-1}=x', u_t)P(X_{t-1}=x'|u_{1:t-1})
$$

## Including observations in the HMM estimate

Now let's say we have a set of observations which provide us some information about the probability (this matrix doesn't have to be square -- we can have more observation modalities than hidden states):

$$
M=
\begin{bmatrix}
P(Y_t=0|X_t=0) & P(Y_t=1|X_t=0) \\
P(Y_t=0|X_t=1) & P(Y_t=1|X_t=1)
\end{bmatrix}
$$

Now our state estimation objective must include the sequence of observations:

$$
P(X_t=x|Y_{1:t})
$$

At this point, I think you get the idea about the notation basically being determined by clarity and preference, so I won't explain myself further. 

If we knew the previous state, we could write $P(X_t=x|X_{t-1},Y_t)$. However, we must marginalize over all possible states of $X_{t-1}$. I will use Bayes Rule in this derivation and the marginalization will appear.

$$
P(X_t=x|Y_{1:t})=\frac{P(X_t=x,Y_{1:t})}{P(Y_{1:t})}=\frac{P(Y_t|X_t=x,Y_{1:t-1})P(X_t=x|Y_{1:t-1})P(Y_{1:t-1})}{P(Y_{1:t})}
$$

Now observe this:

$$
\begin{align*}
P(Y_{1:t})=P(Y_t,Y_{1:t-1})=P(Y_t|Y_{1:t-1})P(Y_{1:t-1})\\
P(Y_{1:t-1})=\frac{P(Y_{1:t})}{P(Y_t|Y_{1:t-1})}
\end{align*}
$$

Replacing this in the original expression:

$$
\begin{align*}
\frac{P(Y_t|X_t=x,Y_{1:t-1})P(X_t=x|Y_{1:t-1})P(Y_{1:t-1})}{P(Y_{1:t})}=\frac{P(Y_t|X_t=x,Y_{1:t-1})P(X_t=x|Y_{1:t-1})P(Y_{1:t})}{P(Y_{1:t})P(Y_t|Y_{1:t-1})}\\
=\frac{P(Y_t|X_t=x,Y_{1:t-1})P(X_t=x|Y_{1:t-1})}{P(Y_t|Y_{1:t-1})}\\
=\frac{P(Y_t|X_t=x)P(X_t=x|Y_{1:t-1})}{P(Y_t|Y_{1:t-1})}\\
=\frac{P(Y_t|X_t=x)\sum_{x'}P(X_t=x,X_{t-1}=x'|Y_{1:t-1})}{P(Y_t|Y_{1:t-1})}\\
=\frac{P(Y_t|X_t=x)\sum_{x'}P(X_t=x|X_{t-1}=x')P(X_{t-1}=x'|Y_{1:t-1})}{P(Y_t|Y_{1:t-1})}\\
\end{align*}
$$

Defining a variable called $\text{bel}_t(x)=P(X_t=x|Y_{1:t})$ will help simplify this expression. I'll also substitute $T$ and $M$:

$$
\begin{align*}
\text{bel}_t(x)=P(X_t=x|Y_{1:t})=\frac{M_{xY_t}\sum_{x'}T_{x'x}\text{bel}_{t-1}(x')}{P(Y_t|Y_{1:t-1})}
\end{align*}
$$

Finally, since the denominator doesn't depend on the state, we can consider this a normalizing factor to simplify. I'll come back to computing this later:

$$
\eta = \frac{1}{P(Y_t|Y_{1:t-1})}
$$

$$
\text{bel}_t(x)=\eta M_{xY_t}\sum_{x'}T_{x'x}\text{bel}_{t-1}(x')
$$


```python

```

## Including observations in the action model estimate

Now let's say we have a set of observations which provide us some information about the state. Maybe the points on the grid are painted either black or white and these values are known ahead of time. The sensor is noisy, correctly predicting the color with an accuracy of .9.

Our new state estimation objective:

$$
\begin{align*}
P(X_t=x|u_{1:t},z_{1:t})=\frac{P(X_t=x,u_{1:t},z_{1:t})}{P(u_{1:t},z_{1:t})}\\
=\frac{P(z_t|X_t=x,u_{1:t},z_{1:t-1})P(X_t=x,u_{1:t},z_{1:t-1})}{P(u_{1:t},z_{1:t})}\\
=\frac{P(z_t|X_t=x,u_{1:t},z_{1:t-1})P(X_t=x|u_{1:t},z_{1:t-1})P(u_{1:t},z_{1:t-1})}{P(u_{1:t},z_{1:t})}\\
=\frac{P(z_t|X_t=x)P(X_t=x|u_{1:t},z_{1:t-1})P(u_{1:t},z_{1:t-1})}{P(u_{1:t},z_{1:t})}\\
\end{align*}
$$

Similar to before, I can calculate the normalizing factor as follows:

$$
\begin{align*}
P(u_{1:t},z_{1:t})=P(z_t|u_{1:t},z_{1:t-1})P(u_{1:t},z_{1:t-1})\\
P(u_{1:t},z_{1:t-1})=\frac{P(u_{1:t},z_{1:t})}{P(z_t|u_{1:t},z_{1:t-1})}\\
\end{align*}
$$

Substituting:

$$
\begin{align*}
\frac{P(z_t|X_t=x)P(X_t=x|u_{1:t},z_{1:t-1})P(u_{1:t},z_{1:t})}{P(z_t|u_{1:t},z_{1:t-1})P(u_{1:t},z_{1:t})}\\
=\frac{P(z_t|X_t=x)P(X_t=x|u_{1:t},z_{1:t-1})}{P(z_t|u_{1:t},z_{1:t-1})}\\
=\frac{P(z_t|X_t=x)\sum_{x'}P(X_t=x,X_{t-1}=x'|u_{1:t},z_{1:t-1})}{P(z_t|u_{1:t},z_{1:t-1})}\\
=\frac{P(z_t|X_t=x)\sum_{x'}P(X_t=x|X_{t-1}=x',u_{1:t},z_{1:t-1})P(X_{t-1}=x'|u_{1:t},z_{1:t-1})}{P(z_t|u_{1:t},z_{1:t-1})}\\
=\frac{P(z_t|X_t=x)\sum_{x'}P(X_t=x|X_{t-1}=x')P(X_{t-1}=x'|u_{1:t},z_{1:t-1})}{P(z_t|u_{1:t},z_{1:t-1})}\\
\end{align*}
$$

Since the denominator doesn't depend on the state, I'll let $\eta=\frac{1}{P(z_t|u_{1:t},z_{1:t-1})}$ once again and substitute:

$$
\begin{align*}
P(X_t=x|u_{1:t},z_{1:t})=\eta P(z_t|X_t=x)\sum_{x'}P(X_t=x|X_{t-1}=x')P(X_{t-1}=x'|u_{1:t},z_{1:t-1})
\end{align*}
$$

Since we don't have a clear definition for the transition (or motion model) and emission (sensor model), I will simply call create two functions and redefine $\text{bel}_t(x)$ in this context:

$$
\begin{align*}
\text{sensor}(z_t, xi) = P(z_t|X_t=x) \\
\text{motion}(x,x')=P(X_t=x|X_{t-1}=x')\\
\text{bel}_t(x)=P(X_t=x|u_{1:t},z_{1:t})
\end{align*}
$$

Finally, I will substite all these into the derived expression:


$$
\begin{align*}
P(X_t=x|u_{1:t},z_{1:t})=\eta \text{sensor}(z_t, x) \sum_{x'}\text{motion}(x,x')\text{bel}_{t-1}(x')
\end{align*}
$$

Finally, I will get to deriving $\eta$. I'm only going to do it for the action model because they're basically the same, and I don't feel like doing the same thing twice!

$$
\begin{align*}
P(z_t|z_{1:t-1})=\sum_x P(z_t|X_t=x,z_{1:t-1})P(X_t=x|z_{1:t-1})\\
=\sum_x P(z_t|X_t=x)P(X_t=x|z_{1:t-1})\\
=\sum_x P(z_t|X_t=x)\sum_{x'}P(X_t=x|X_{t-1}=x',z_{1:t-1})P(X_{t-1}=x'|z_{1:t-1})\\
=\sum_x P(z_t|X_t=x)\sum_{x'}P(X_t=x|X_{t-1}=x')P(X_{t-1}=x'|z_{1:t-1})\\
=\sum_x \text{sensor}(z_t,x)\sum_{x'}\text{motion}(x,x')\text{bel}_{t-1}(x')\\
\end{align*}
$$

I will include everything for a single Bayes filter in one equation because it's satisfying to look at. This means I'll have to rename variables to avoid overlap.

$$
\begin{align*}
P(X_t=x|u_{1:t},z_{1:t})=\frac{\text{sensor}(z_t, x) \sum_{x'}\text{motion}(x,x')\text{bel}_{t-1}(x')}{\sum_{x''} \text{sensor}(z_t,x'')\sum_{x'''}\text{motion}(x'',x''')\text{bel}_{t-1}(x''')\\}
\end{align*}
$$

This is nice. Next stop, Kalman Filters. After that, Extended and Unscented Kalman Filters.
