---
title: "An Introduction to Diffusion Models"
date: 2022-10-27T16:26:37-07:00
draft: false
math: true
toc: true
---

## Introduction

*Denoising Diffusion Probabilisitc Models* ("DDPMs" or just "Diffusion models") have garnered significant interest as of late due to their use in the recent slate of amazing text-to-image models such as DALL·E 2 and Stable Diffusion. In this post I will explain the basic mechanisms and math behind how these models work.

DDPMs ([Sohl-Dickstein et al., 2015](https://arxiv.org/pdf/1503.03585.pdf)) are probabilistic generative models, meaning that they aim to learn an approximation for a target data distribution $q(x^{(0)})$. This can be done for a number of reasons, for example to draw new samples from that distribution (new images similar to training images, new text similar to training corpus), or to learn a latent variable representation of the data in an unsupervised fashion. Flexible probabilistic models meant to model a wide range of distributions are frequently intractable due to issues such as computing the *partition function* $\(\int \tilde{p}(x) dx\)$ for an unnormalized probability distribution $\tilde{p}(x)$. A common approach to circumvent these issues is to instead learn a data generation process and follow this process to draw new samples. This is the approach taken by DDPMs.

## Data Generation Process

DDPMs model the data generation process as the inverse of a *forward diffusion process*, which is a Markov chain that converts the initial data distribution $q(x^{(0)})$ into a tractable prior $\pi(y)$ via repeated application of a *diffusion kernel* $$q(x^{(t)} | x^{(t-1)}) = T_t(x^{(t)} | x^{(t-1)}, \beta_t)$$

For the remainder of this post I will assume the standard choices
$$
\begin{aligned} \pi(y) &\sim \mathcal{N}(y; \textbf{0}, \textbf{I}) \cr
q(x^{(t)} | x^{(t-1)}) &\sim \mathcal{N}(x^{(t)}; \sqrt{1 - \beta_t}x^{(t-1)}, \beta_t\textbf{I})
\end{aligned}
$$

though for same target data distributions (e.g. discrete ones) this may not be appropriate. This choice of prior and kernel has the convenient property that the intermediate distributions of the noised data $q(x^{(t)} | x^{(0)})$ can be calculated in closed form, without running the Markov chain for $t$ steps:
$$
\begin{aligned}
q(x^{(t)} | x^{(0)}) &\sim \mathcal{N}(x^{(t)}; \sqrt{\tilde{\alpha}_t}x^{(0)}, \sqrt{1 - \tilde{\alpha}_t}\textbf{I}) \cr
\text{with } \tilde{\alpha}_t &:= \prod\_{i=0}^t (1 - \beta_i)
\end{aligned}
$$

Note that for suitably chosen diffusion parameters $\beta_t$, as the Markov chain length $T \rightarrow \infty$, then $q(x^{(T)} | x^{(0)})$ converges in distribution to the prior $\pi(y)$ as desired.

Based on this forward process, the data generation process is conceptually straightforward. We run the Markov chain in reverse, starting with a prior sample $x^{(T)} \sim \pi(x^{(T)})$, then repeatedly drawing samples from the inverted diffusion kernels $q(x^{(t-1)} | x^{(t)})$ for $t = T ... 1$ until reaching a final sample $x^{(0)}$. Of course, the exact inverted kernels $q(x^{(t-1)} | x^{(t)})$ is unavailable to us, so we must approximate them with a learned distribution $p_{\theta}(x^{(t-1)} | x^{(t)}) \approx q(x^{(t-1)} | x^{(t)})$. For small $\beta_t$, $q(x^{(t-1)} | x^{(t)})$ is also normal, meaning that we can learn $p_{\theta}$ by learning the distribution mean $\mu_{\theta}(x^{(t)}, t)$ and variance $\Sigma_{\theta}(x^{(t)}, t)$ with a function approximator such as a neural network.

## Loss Function

As the negative log likelihood $\mathbb{E}[-\text{log }p_{\theta}(x^{(0)})]$ is intractable, we train in standard fashion by optimizing the *variational lower bound*
$$
\mathbb{E}\_{q(x^{(1...T)} | x^{(0)})} \left[ -\text{log }\frac{p\_{\theta}(x^{(0...T)})}{q(x^{(1...T)} | x^{(0)})} \right] \geq -\text{log }p\_{\theta}(x)
$$
I will now give the main results related to the DPPM loss function, following the calculations in [Luo (2022)](https://arxiv.org/pdf/2208.11970.pdf). Expanding and using the Markov property $q(x^{(t)} | x^{(t-1)}) = q(x^{(t)} | x^{(t-1)}, x^{(0)})$ yields the loss:
$$
\begin{aligned}
L\_{VLB} := \text{ } &\mathbb{E}\_{q(x^{(1)} | x^{(0)})} \left[ \text{log } p\_{\theta}(x^{(0)} | x^{(1)}) \right] \cr
&- \sum_{t = 2}^T \mathbb{E}\_{q(x^{(t)} | q^{(0)})} \left[ D\_{KL}(q(x^{(t-1)} | x^{(t)}, x^{(0)}) || p\_{\theta}(x^{(t-1)} | x^{(t)})) \right]
\end{aligned}
$$
The individual terms in the summation are referred to as $L_t$ in the literature and in the rest of this post.
Conditioning the inverted diffusion kernel on $x^{(0)}$ surprisingly gives it the tractable form
$$
\begin{aligned}
    q(x^{(t-1)} | x^{(t)}, x^{(0)}) &= \mathcal{N}(x^{(t-1)}; \tilde{\mu}_t(x^{(t)}, x^{(0)}), \tilde{\beta}_t\textbf{I}) \cr
    \text{where } \tilde{\mu}_t(x^{(t)}, x^{(0)}) &:= \frac{\sqrt{\tilde{\alpha}\_{t-1}}\beta_t}{1 - \tilde{\alpha_t}}x^{(0)} + \frac{\sqrt{1 - \beta_t}(1 - \tilde{\alpha}\_{t-1})}{1 - \tilde{\alpha_t}}x^{(t)} \cr
    \text{and } \tilde{\beta}_t &:= \frac{1 - \tilde{\alpha}\_{t-1}}{1 - \tilde{\alpha}_t}\beta_t
\end{aligned}
$$
meaning that each of the terms in the summation in $L$ are the divergence between two Gaussians and therefore easily computable, taking the form:
$$
D\_{KL}(q(x^{(t-1)} | x^{(t)}, x^{(0)}) || p\_{\theta}(x^{(t-1)} | x^{(t)})) = \frac{1}{2 \sigma^2_q(t)}|| \tilde{\mu}_t(x^{(t)}, x^{(0)}) - \mu\_{\theta}(x^{(t)}, t)||^2
$$
where we are assuming that the approximated variance $\Sigma\_{\theta}(x^{(t)}, t) = \sigma^2_q(t) \textbf{I}$ is isotropic and varies only with $t$. Given the exact form of $\tilde{\mu}_t(x^{(t)}, x^{(0)})$ above, we can parameterize our neural network to learn an approximation of the ground truth image $\hat{x}\_{\theta}(x^{(t)}, t) \approx x^{(0)}$, as
$$
\begin{aligned}
    D\_{KL}(q(x^{(t-1)} | x^{(t)}, x^{(0)}) || p\_{\theta}(x^{(t-1)} | x^{(t)})) = \frac{1}{2 \sigma^2_q(t)} \frac{\tilde{\alpha}\_{t-1}\beta_t^2}{(1 - \tilde{\alpha}_t)^2} || x^{(0)} - \hat{x}\_{\theta}(x^{(t)}, t)||^2
\end{aligned}
$$

Alternatively, following [Ho et al. (2020)](https://arxiv.org/pdf/2006.11239.pdf), the neural network can learn to predict the specific noise $\epsilon \sim \mathcal{N}(\textbf{0}, \textbf{I})$ that corrupted $x^{(t)}$ from $x^{(0)}$. In fact, they fix the variances $\sigma^2_q(t)$ and learn using a reweighted objective:
$$
L\_{simple} = \mathbb{E}\_{t, x_0, \epsilon} \left[ ||\epsilon - \hat{\epsilon}\_{\theta}(x^{(t)}, t) ||^2 \right]
$$
which in practice led to more stable learning. The reweighted objective underweights the loss from the initial noising steps and overweights the loss from the later ones. Intuitively, this improves training because the initial steps add very little noise compared to the later ones and are as a result easier to learn. This is the formulation of the loss function most commonly used in practice. In either form, the DDPM can be seen as learning a sequence of denoising autoencoders.

## Improvements

In this section, I will cover a couple of interesting improvements to the DDPM baseline that [Nichol \& Dhariwal (2021)](https://arxiv.org/pdf/2102.09672.pdf) made. DDPMs had up to that paper yielded high-quality samples in a number of domains, such as image and audio generation, but were yet to produce competitive log-likelihoods when compared to more established generative model classes such as Variational Autoencoders. This suggested that they had poor mode coverage in comparison.

### Learning the Variance
Note that the reweighted objective $L\_{simple}$ is constant with respect to the approximated variance, forcing it to be fixed prior to training. Ho et al. noticed that this produced high sample quality across a range of plausible variances (from $\beta_t$ to $\tilde{\beta}_t$), while learning the variance led to unstable learning. Nichol \& Dhariwal instead learned a variance interpolated in the plausible range:
$$
\Sigma\_{\theta}(x^{(t)}, t) = \text{exp} (v \text{ log }\beta_t + (1 - v) \text{ log } \tilde{\beta}_t )
$$
where $v$ is a vector outputted by the model. As $L\_{simple}$ cannot guide the training of $v$, they use a hybrid objective
$$
L\_{hybrid} = L\_{simple} + \lambda L\_{VLB}
$$
with $\lambda = 0.0001$. Because they intended for the new $L\_{VLB}$ term to only guide the choice of variance, they stopped the gradient for that term from flowing to the mean approximation $\mu\_{\theta}(x^{(t)}, t)$. They observed that $L\_{VLB}$ is quite noisy, making optimization difficult. To reduce variance, they resampled the individual terms in $L\_{VLB}$ with importance sampling:
\begin{align*}
    L\_{resampled} = \mathbb{E}\_{t \sim p_t} \left[ \frac{L_t}{p_t} \right] \\
    \text{where } p_t \propto \sqrt{\mathbb{E} \left[ L_t^2 \right]}
\end{align*}

### Noise Schedule
Nichol \& Dhariwal also noticed that the linear noise schedule used in Ho et al. seemed to corrupt the samples too quickly, as a large chunk of the transitions at the end of the Markov chain could be removed with little to no drop in sample quality. Instead they used a cosine noise schedule:
$$
\begin{aligned}
    \tilde{\alpha}_t &= \frac{f(t)}{f(0)} \\
    \text{where } f(t) &:= \text{cos } \left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)
\end{aligned}
$$
with $s$ a small constant added to produce sufficient noise in the early transitions.


### Strided Ancestral Sampling
In the baseline DDPM, sampling is extremely slow, as it entails running inference through the trained network once per denoising step for each new sample. The state of the art DDPMs frequently contained thousands of denoising steps, meaning that a single sample could take minutes to generate on a GPU. Nichol \& Dhariwal propose a simple solution to this problem, which is to sample using a subsequence of the denoising steps. They reduce $T$ steps to $K$ by picking $K$ evenly spaced time steps and running the reverse diffusion process across those only. This enabled near-optimal metrics with up to $40$ times fewer denoising steps.

## Conditioned Diffusion Models
A natural extension to generatively modeling a data distribution $p(x)$ is to model a conditional distribution $p(x | y)$. Some example applications of this formulation are upsampling models which output a high resolution image conditioned on a low resolution version of the same image, or image synthesizers that output an image conditioned on class information or text.

A naive approach to this problem is to simply pass in the conditioning information as input to the network along with the noised samples and time step. This is common and can yield good results, but two main strategies building on this approach have been used to even greater success: classifier guidance and classifier-free guidance.

### Classifier Guidance
Logically, to train with classifier guidance we need a classifier $p_{\phi}(y | x^{(t)})$ which predicts the conditioning information. Note that even if we are modeling a dataset which has highly accurate pretrained classifiers available, such as ImageNet or CIFAR-10, these will most likely be insufficient for guidance due to poor performance on \textit{noised} samples.

Given such a classifier, [Dhariwal \& Nichol (2021)](https://arxiv.org/pdf/2105.05233.pdf) derive an approximate form for the conditioned reverse sampling distribution:
$$
p\_{\theta, \phi}(x^{(t)} | x^{(t-1)}, y) \approx \mathcal{N}(x^{(t)}; \mu\_{\theta}(x^{(t)}, t) + w\Sigma\_{\theta}(x^{(t)}, t)\nabla_{x^{(t)}} \text{log } p\_{\phi}(y | x^{(t)}), \Sigma_\{\theta}(x^{(t)}, t))
$$
Based on this form, the surprising result is that the conditioning only has to factor in at the sampling stage. By training an unconditioned diffusion model $\mu\_{\theta}, \Sigma\_{\theta}$ and a noisy classifier $p_{\phi}$, conditioned samples can be generated by moving the sampling mean in the direction of the classifier gradient. A scaling term $w$ has been added to govern how much classifier guidance is desired. Empirically, increasing the amount of classifier guidance trades off sample quality for sample diversity, with increasing guidance decreasing diversity.

### Classifier-Free Guidance
Classifier-Free Guidance ([Ho \& Salimans, 2021](https://openreview.net/pdf?id=qw8AKxfYbI)) is another method that can be used to tune the exact amount of conditioning information used in the sampling process. A problem with Classifier Guidance is that it requires the separate training of a noised sample classifier alongside the diffusion model. Instead of using a classifier, Classifier-Free Guidance uses both an unconditioned diffusion model $p\_{\theta}(x^{(t-1)} | x^{(t)}, t)$ and a conditioned diffusion model $p\_{\phi}(x^{(t-1)} | x^{(t)}, t, y)$. The exact distribution used for sampling is then
$$
\tilde{p}(x^{(t-1)} | x^{(t)}, t, y) = wp\_{\theta}(x^{(t-1)} | x^{(t)}, t) + (1-w)p\_{\phi}(x^{(t-1)} | x^{(t)}, t, y)
$$
where once again $w$ is a hyperparameter that scales the exact amount of guidance used in the sampling process. This parameter displays similar control of the sample quality-diversity trade-off as its analogue in Classifier Guidance.

At first glance this seems to have the same problem as Classifier Guidance, as it looks like it requires training two distinct diffusion models. However this can be cleverly avoided by instead training a singular conditional diffusion model and randomly dropping out the conditioning information during training.

### Comparing Classifier and Classifier-Free Guidance
In large-scale systems (eg DALL·E 2, Imagen) based on conditioned diffusion models, Classifier-Free Guidance is almost exclusively used. A head to head comparison of both conditioning methods was performed in the development of GLIDE ([Nichol et al., 2022](https://arxiv.org/pdf/2112.10741.pdf)), a text-to-image model from OpenAI. Besides the obvious advantage that Classifier-Free Guidance requires training only one model instead of two, the GLIDE authors additionally note that it empirically outperforms Classifier Guidance in photorealism and text-image alignment according to human evaluation.

## References
Dhariwal, P., \& Nichol, A. (2021). Diffusion models beat GANs on image synthesis. In arXiv [cs.LG]. http://arxiv.org/abs/2105.05233

Ho, J., Jain, A., \& Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In arXiv [cs.LG]. http://arxiv.org/abs/2006.11239

Ho, J., \& Salimans, T. (2022). Classifier-Free Diffusion Guidance. In arXiv [cs.LG]. http://arxiv.org/abs/2207.12598

Luo, C. (2022). Understanding diffusion models: A unified perspective. In arXiv [cs.LG]. http://arxiv.org/abs/2208.11970

Nichol, A., \& Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In arXiv [cs.LG]. http://arxiv.org/abs/2102.09672

Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., \& Chen, M. (2021). GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In arXiv [cs.CV]. http://arxiv.org/abs/2112.10741

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., \& Chen, M. (2022). Hierarchical text-conditional image generation with CLIP latents. In arXiv [cs.CV]. http://arxiv.org/abs/2204.06125

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S. K. S., Ayan, B. K., Mahdavi, S. S., Lopes, R. G., Salimans, T., Ho, J., Fleet, D. J., \& Norouzi, M. (2022). Photorealistic text-to-image diffusion models with deep language understanding. In arXiv [cs.CV]. http://arxiv.org/abs/2205.11487

Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., \& Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In arXiv [cs.LG]. http://arxiv.org/abs/1503.03585
