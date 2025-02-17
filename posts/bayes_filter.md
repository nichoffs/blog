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

We could equivalently define $X_t$ in terms of $X_{t-1}$ and our transition model, which makes the relevance of recursion a bit clearer.

$$P(X_t=x|X_0,T)=P(X_t=x|X_{t-1}, T)=P(X_t=x|X_{t-1})$$

The final equivalency is just to show that $T$ is usually not explicitly included in the probability.

At this point, a recursive definition that allows for massive reduction in computational complexity comes very naturally. Let's define a quantity $\alpha(X_t=x)=P(X_t=x|X_{t-1})$, or alternatively $\alpha_t(x)$ (I'll use the latter notation). Now, we can write the above equation as follows:

$$
\alpha_t(x)=P(X_t=x|X_{t-1})=\sum_{x'} P(X_t=x|X_{t-1}=x')P(X_{t-1}=x'|X_{t-2})=\sum_{x'} P(X_t=x|X_{t-1}=x')\alpha_{t-1}(x')
$$

Using our transition matrix:

$$
\alpha_t(x)=\sum_{x'} P(X_t=x|X_{t-1}=x')\alpha_{t-1}(x')=\sum_{x'} T_{x'x}\alpha_{t-1}(x')
$$

To get $P(X_t|X_{t-1})$, we must calculate $\alpha_t(x)$ for all possible state values $x$.


## Action Model -- Transitioning between states without observation

In the action model, our state is the position of a robot in a $2 \times 2$ grid. We are given a list of actions $u_{1:t}$ over the timesteps along with probabilities related to how an action affects the state. This is known as the **motion model**, similar to $T$ in the hidden markov model. A robot can either move up, down, left, or right. It succeeds with probability $.9$ and fails with probability $.1$. If the action would move the robot off the grid, it stays in place with absolute certainty ($1$).

I think this approach lends itself more to a code implementation, although I won't get into details in this post.

If our set of actions is $u_{1:t}$, then our probability of interest is the state at time $t$:

$$
P(X_t=x|X_0, u_{1:t})
$$

Similarly to the HMM, this can be equivalently expressed in terms of the previous state and the current action.

$$
P(X_t=x|X_0, u_{1:t})=P(X_t=x|X_{t-1}, u_t)
$$

The same notational style as the HMM is applied here. From here, we can use the law of total probability:


$$
P(X_t=x|X_{t-1}, u_t)=\sum_{x'} P(X_t=x|X_{t-1}=x', u_t)P(X_{t-1}=x'|X_{t-2}, u_{t-1})
$$

I'll define a variable $\text{bel}(X_t=x)$ or alternatively $\text{bel}_t(x)$ which makes the recursion clearer:

$$
\text{bel}_t(x)=P(X_t=x|X_{t-1}, u_t)=\sum_{x'} P(X_t=x|X_{t-1}=x', u_t)P(X_{t-1}=x'|X_{t-2}, u_{t-1})=\sum_{x'} P(X_t=x|X_{t-1}=x', u_t)\text{bel}_{t-1}(x')
$$

## Comparing the two approaches

I'll put the HMM and motion model equations side by side:

$$
\begin{array}{cc}
\alpha_t(x) = \sum_{x'} P(X_t = x | X_{t-1} = x') \alpha_{t-1}(x') & 
\text{bel}_t(x) = \sum_{x'} P(X_t = x | X_{t-1} = x', u_t) \text{bel}_{t-1}(x')
\end{array}
$$

See the similarities? That's because they're really doing the same thing!


## Including observations in the HMM estimate

Now let's say we have a set of observations which provide us some information about the probability (this matrix doesn't have to be square -- we can have more observation modalities than hidden states):

$$
M=
\begin{bmatrix}
P(Y_t=0|X_t=0) & P(Y_t=1|X_t=0) \\
P(Y_t=0|X_t=1) & P(Y_t=1|X_t=1)
\end{bmatrix}
$$

Now our state estimation objective must include the sequence of observations (existence of $T$ and $M$ are assumed):

$$
P(X_t=x|X_0, Y_{1:t})=P(X_t=x|X_{t-1}, Y_t)
$$

Using Bayes Rule, I can expand this out:

$$
\begin{align*}
P(X_t=x|X_{t-1}, Y_t) = \frac{P(X_t=x,X_{t-1},Y_t)}{P(X_{t-1}, Y_t)} \\ 
=\frac{P(Y_t|X_{t}=x, X_{t-1})P(X_t=x, X_{t-1})}{P(X_{t-1}, Y_t)} \\
=\frac{P(Y_t|X_{t}=x, X_{t-1})P(X_t=x, X_{t-1})}{P(X_{t-1}, Y_t)} \\
=\frac{P(Y_t|X_{t}=x)P(X_t=x, X_{t-1})}{P(X_{t-1}, Y_t)} \\
=\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})P(X_{t-1})}{P(X_{t-1}, Y_t)} \\
=\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})P(X_{t-1})}{P(Y_t|X_{t-1})P(X_{t-1})} \\
=\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})}{P(Y_t|X_{t-1})} \\
\end{align*}
$$

As you can tell, the denominator $P(Y_t|X_{t-1})$ is a bit weird. An observation at $Y_t$ is only dependent on $X_t$, so we can actually just write $P(Y_t|X_{t-1})$ as $P(Y_t)$: 

$$
\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})}{P(Y_t|X_{t-1})}
$$

$P(Y_t)$, using the notational style I described earlier, can be expressed as $P(Y_t|X_t)$. Using the law of total probability:

$$
\begin{align*}
\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})}{P(Y_t|X_{t-1})} \\
=\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})}{P(Y_t|X_t)} \\
=\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})}{\sum_x P(Y_t|X_t=x)P(X_t=x)} \\
=\frac{P(Y_t|X_{t}=x)P(X_t=x|X_{t-1})}{\sum_{x'} P(Y_t|X_t=x')P(X_t=x'|X_{t-1})}
\end{align*}
$$

Now, I will redefine the variable $\alpha_t(x)$ I defined earlier:

$$
\alpha_t(x)=P(X_t=x|X_{t-1},Y_t)
$$

The derivation for this is going to be huge because I want to don't want to skip any steps. You can go line by line and see which rules are used, but it's a lot of Markov assumption, law of total probability, observation independence. Same things as always. With that preface, here we go.

$$
\begin{align*}
\alpha_t(x)=P(X_t|X_{t-1},Y_t)=\frac{P(X_t=x,X_{t-1},Y_t)}{P(X_{t-1},Y_t)} \\
= \frac{P(Y_t|X_t)P(X_t=x,X_{t-1})}{P(X_{t-1},Y_t)} \\
= \frac{P(Y_t|X_t=x)P(X_t=x|X_{t-1})P(X_{t-1})}{\sum_{x'}P(X_t=x',X_{t-1},Y_t)} \\
= \frac{P(Y_t|X_t=x)P(X_t=x|X_{t-1})P(X_{t-1}|X_{t-2}, Y_{t-1})}{\sum_{x'}P(X_t=x',X_{t-1},Y_t)} \\
= \frac{P(Y_t|X_t=x)P(X_t=x|X_{t-1})P(X_{t-1}|X_{t-2}, Y_{t-1})}{\sum_{x'}P(Y_t,X_t=x'|X_{t-1})P(X_{t-1}|X_{t-2},Y_{t-1})} \\
= \frac{P(Y_t|X_t=x)P(X_t=x|X_{t-1})P(X_{t-1}|X_{t-2}, Y_{t-1})}{\sum_{x'}\sum_{x'''}P(Y_t,X_t=x'|X_{t-1}=x''')P(X_{t-1}=x'''|X_{t-2},Y_{t-1})} \\
= \frac{P(Y_t|X_t=x)P(X_t=x|X_{t-1})P(X_{t-1}|X_{t-2}, Y_{t-1})}{\sum_{x'}P(Y_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''')P(X_{t-1}=x'''|X_{t-2},Y_{t-1})} \\
= \frac{P(Y_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'')P(X_{t-1}=x''|X_{t-2}, Y_{t-1})}{\sum_{x'}P(Y_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''')P(X_{t-1}=x'''|X_{t-2},Y_{t-1})} \\
\end{align*}
$$

Finally, that was a lot. Now we can substitute $\alpha$:

$$
\frac{P(Y_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'')P(X_{t-1}=x''|X_{t-2}, Y_{t-1})}{\sum_{x'}P(Y_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''')P(X_{t-1}=x'''|X_{t-2},Y_{t-1})}=\frac{P(Y_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'')\alpha_{t-1}(x'')}{\sum_{x'}P(Y_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''')\alpha_{t-1}(x''')}
$$

Substituting with $T$ and $M$, we get the final formula:

$$
\alpha_t(x)=\frac{M_{xY_t}\sum_{x''}T_{x''x}\alpha_{t-1}(x'')}{\sum_{x'}M_{x'Y_t}\sum_{x'''}T_{x'''x}\alpha_{t-1}(x''')}
$$

Since the denominator doesn't depend on $x$, people usually take it out and call it $\eta$. 

For the following $\eta$:

$$
\eta = \frac{1}{\sum_{x'}M_{x'Y_t}\sum_{x'''}T_{x'''x}\alpha_{t-1}(x''')}
$$

We can rewrite $\alpha_t(x)$ as follows:


$$
\alpha_t(x)=\eta M_{xY_t}\sum_{x''}T_{x''x}\alpha_{t-1}(x'')
$$

Beautiful!

## Include observations in the motion model state

So imagine each cell is either colored black or white and we know this ahead of time. We also have a sensor which can detect the color with an accuracy of $.9$. Call the list of observations $z_{1:t}$. Now our state estimation should include this observation, as it provides us some information on the hidden state. At some point in the derivation, I will replace $P(X_{t-1},u_t)$ with $P(X_{t-1}|X_{t-2},u_{t-1},z_{t-1}). I can do this because $u_t$ is already known, $X_{t-1}$ is not conditionally dependent on $u_t$, and the notation from earlier lets me write $P(X_{t-1})$ as $P(X_{t-1})={(X_{t-1}|X_{t-2},u_{t-1},z_{t-1})}. Therefore, $P(X_{t-1}|u_t)P(u_t)=P(X_{t-1})=P(X_{t-1}|X_{t-2},u_{t-1},z_{t-1})$. With that being said, here's the derivation:

$$
\begin{align*}
P(X_t=x|X_{t-1},u_t,z_t) = \frac{P(X_t=x,X_{t-1},u_t,z_t)}{P(X_{t-1},u_t,z_t)} \\
=\frac{P(z_t|X_t=x, X_{t-1}, u_t)P(X_t=x,X_{t-1},u_t)}{P(X_{t-1},u_t,z_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x,X_{t-1},u_t)}{P(X_{t-1},u_t,z_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1}, u_t)P(X_{t-1},u_t)}{P(X_{t-1},u_t,z_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1},u_t)P(X_{t-1},u_t)}{P(X_{t-1},u_t,z_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1},u_t)P(X_{t-1},u_t)}{P(z_t|X_{t-1},u_t)P(X_{t-1},u_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1},u_t)P(X_{t-1},u_t)}{\sum_{x'}P(z_t,X_t=x'|X_{t-1},u_t)P(X_{t-1},u_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1},u_t)P(X_{t-1},u_t)}{\sum_{x'}P(z_t|X_t=x',X_{t-1},u_t)P(X_t=x'|X_{t-1},u_t)P(X_{t-1},u_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1},u_t)P(X_{t-1},u_t)}{\sum_{x'}P(z_t|X_t=x',X_{t-1},u_t)P(X_t=x'|X_{t-1},u_t)P(X_{t-1},u_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1},u_t)P(X_{t-1},u_t)}{\sum_{x'}P(z_t|X_t=x')P(X_t=x'|X_{t-1},u_t)P(X_{t-1},u_t)} \\
=\frac{P(z_t|X_t=x)P(X_t=x|X_{t-1},u_t)P(X_{t-1}|X_{t-2},u_{t-1},z_{t-1})}{\sum_{x'}P(z_t|X_t=x')P(X_t=x'|X_{t-1},u_t)P(X_{t-1}|X_{t-2},u_{t-1},z_{t-1})} \\
=\frac{P(z_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'',u_t)P(X_{t-1}=x''|X_{t-2},u_{t-1},z_{t-1})}{\sum_{x'}P(z_t|X_t=x')P(X_t=x'|X_{t-1},u_t)P(X_{t-1}|X_{t-2},u_{t-1},z_{t-1})} \\
=\frac{P(z_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'',u_t)P(X_{t-1}=x''|X_{t-2},u_{t-1},z_{t-1})}{\sum_{x'}P(z_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''',u_t)P(X_{t-1}=x'''|X_{t-2},u_{t-1},z_{t-1})} \\
\end{align*}
$$

This is absolutely insane, but now I can simplify by redefining with the variable $\text{bel}_t(x)=P(X_t=x|X_{t-1},u_t,z_t)$:

$$
\frac{P(z_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'',u_t)P(X_{t-1}=x''|X_{t-2},u_{t-1},z_{t-1})}{\sum_{x'}P(z_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''',u_t)P(X_{t-1}=x'''|X_{t-2},u_{t-1},z_{t-1})} = \frac{P(z_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'',u_t)\text{bel}_{t-1}(x'')}{\sum_{x'}P(z_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''',u_t)\text{bel}_{t-1}(x''')}
$$

Since the denominator doesn't depend on $x$, we can take it out as $\eta$:

$$
\eta=\frac{1}{\sum_{x'}P(z_t|X_t=x')\sum_{x'''}P(X_t=x'|X_{t-1}=x''',u_t)\text{bel}_{t-1}(x''')}
$$

Rewriting the expression:

$$
\text{bel}_t(x)=\eta P(z_t|X_t=x)\sum_{x''}P(X_t=x|X_{t-1}=x'',u_t)\text{bel}_{t-1}(x'')
$$

I don't have as simple a way to represent the motion model or sensor model, but I'll define a function called $\text{motion}$ and $\text{sensor}$. $\text{motion}()$:

$$
\begin{aligned}
\text{motion}(x, x', u_t) &= P(X_t = x \mid X_{t-1} = x', u_t),\\[1mm]
\text{sensor}(x, z_t) &= P(z_t \mid X_t = x).
\end{aligned}
$$

Then, using the definition
$$
\text{bel}_t(x) = P(X_t=x\mid X_{t-1},u_t,z_t),
$$

we can write the belief update as
$$
\text{bel}_t(x)=\eta\, \text{sensor}(x, z_t)\sum_{x'} \text{motion}(x, x', u_t)\, \text{bel}_{t-1}(x'),
$$

where the normalization constant $\eta$ is given by

$$
\eta = \frac{1}{\sum_{x''}\text{sensor}(x'', z_t)\sum_{x'} \text{motion}(x'', x', u_t)\, \text{bel}_{t-1}(x')}.
$$

I think this interpretation is pretty intuitive.

## Comparing the two approaches

 
$$
\begin{array}{cc}
\alpha_t(x)=\frac{M_{xY_t}\sum_{x''}T_{x''x}\alpha_{t-1}(x'')}{\sum_{x'}M_{x'Y_t}\sum_{x'''}T_{x'''x}\alpha_{t-1}(x''')} & \text{bel}_t(x) = \frac{\text{sensor}(x, z_t)\sum_{x'} \text{motion}(x, x', u_t)\, \text{bel}_{t-1}(x')}{\sum_{x''}\text{sensor}(x'', z_t)\sum_{x'} \text{motion}(x'', x', u_t)\, \text{bel}_{t-1}(x')}
\end{array}
$$

The same!
