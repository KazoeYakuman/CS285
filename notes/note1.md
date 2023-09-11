# What is Reinforcement Learning?

- Mathematical formalism for learning-based decision making.

- Approach for learning decision making and control from experience.

**How is this different from other machine learning topics?**

- Standard (supervised) machine learning
  
  - Given $(\mathbf x_i,y_i)$, learn to predict $y$ from $\mathbf x$.
  
  - Usually assumes: i.i.d. data; known ground truth outputs in training.

- Reinforcement learning
  
  - Data is not i.i.d.: previous outputs influence future inputs.
  
  - Ground truth answer is not known, only know if we succeeded or failed.
    
    - More generally, we know the reward.

---

# Supervised Learning of Behaviors

#### Terminology & notation

- $\mathbf o_t$: observation

- $\mathbf s_t$ or $\mathbf x_t$: state

- $\mathbf a_t$ or $\mathbf u_t$: action

- $\pi_\theta(\mathbf a_t| \mathbf o_t)$: policy

- $\pi_\theta(\mathbf a_t|\mathbf s_t)$: policy (fully observed)

## Imitation Learning

### Behavioral cloning

Regard $(\mathbf o_t, \mathbf a_t)$ as training data. Using supervised learning to get $\pi_\theta(\mathbf a_t|\mathbf o_t)$.

Unfortunately, it doesn't work. ([ALVINN: Autonomous Land Vehicle In a Neural Network, 1989](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)) 

Little mistake will lead it to a different state from training data. Once it see something that is a little different, it is more likely to make a slightly bigger mistake. Each additional mistake puts it in a state that is more and more unfamiliar. By the end it will make extremely large mistakes.

However, it works in [End to End Learning for Self-Driving Cars, Bojarski et al. 2016, NVIDIA](https://arxiv.org/abs/1604.07316). In this paper, the car has three cameras. The regular forward pacing camera is the one that is actually going to be driving the car. It takes images from the left facing camera and label them not with the steering command that the human actually executed during data collection but with a modified steering command that steers a little bit to the right.

##### Ideas

- Imitation learning via behavior cloning is not guaranteed to work.
  
  - This is different from supervised learning.
  
  - The reason: i.i.d. assumption does not hold!

- We can formalize why this is and do a bit of theory.

- We can address the problem in a few ways:
  
  - Be smart about how we collect (and augment) our data.
  
  - Use very powerful models that make very few mistakes.
  
  - Use multi-task learning.
  
  - Change the algorithm (DAgger)

**The distributional shift problem:** 

We train under $p_{data}(\mathbf o_t)$, $\max E_{\mathbf o_t \sim p_{data(\mathbf o _t)}}[\log \pi_\theta(\mathbf a_t|\mathbf o_t)].$

We test under $p_{\pi_\theta}(\mathbf o_t)$, $p_{data}(\mathbf o_t) \ne p_{\pi_\theta}(\mathbf o_t)$.

**Define more precisely what we want:**

Define a cost function:

$$
c(\mathbf s_t, \mathbf a_t) =
\begin{cases}
0\ \text{if}\  \mathbf a_t = \pi^*(\mathbf s_t) \\ 
1\ \text{if}\  \text{otherwise}
\end{cases}
$$

Where $\pi^*$ is human driver's policy. This cost function means the number of mistakes the policy makes when we run it.

We assume that there is an small $\epsilon$ such that:

$$
\pi_\theta (\mathbf a \ne \pi^*(\mathbf s)|\mathbf s) \le \epsilon,\  
\text{for all} \  \mathbf s \in \mathcal D_{train} 
$$

In the first time step, the probability of making a mistake is $\epsilon$. Once you make a mistake, you are expected to always make mistakes since the mistake you made before will lead you to a unfamiliar environment. Consider the following time steps, we can get a bound on  the total cost:

$$
\mathbb E[\sum_t c(\mathbf s_t, \mathbf a_t)] \le \epsilon T
+(1-\epsilon)(\epsilon(T-1)+\cdots)
$$

Since $\epsilon$ is very small, the $1-\epsilon$ is close to $1$. The right hand side has $T$ terms, each term is $O(\epsilon T)$. So the expectation of the number of mistakes we will make is $O(\epsilon T^2)$. We can see why behavior cloning doesn't always work: if $T$ is large enough, the algorithm will make a lot of mistakes.

In reality, we can often recover from mistakes. But that doesn't mean that imitation learning will allow us to learn how to do that.

> A paradox: Imitation learning can work better if the data has more mistakes (and recoveries) !

#### Why might we fail to fit the expert?

- Non-Markovian behavior
  
  - Use the whole history (But it might make causal confusion and other problems :( )

- Multimodal behavior
  
  - More expressive continuous distributions
    
    - Mixture of Gaussians
    
    - Latent variable models
    
    - Diffusion models
  
  - Discretization with high-dimensional action spaces
    
    - It is great for 1d actions, but in higher dimensions, discretizing the full space is impractical. **Solution:** discretize one dimension at a time.
    
    - 
