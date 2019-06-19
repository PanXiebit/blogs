## Training Language GANs from Scratch
### Motivation
我们知道language GAN非常难训练，主要是因为gradient estimation, optimization instability, and mode collapse等原因，这导致很多NLPer选择先基于maximum likelihood对模型进行预训练，然后在用language GAN进行fine-tune.作者认为这种 fine-tune 给模型带来的benefit并不clear，甚至会带来不好的效果。  

> 关于mode collapse，李宏毅老师讲过，在对话生成时，模型总是倾向于生成“我不知道”，”我知道了”这样通用的没有太多sense的回复，其实就是属于mode collapse. 类似于图像领域，既要生成鼻子，又要生成嘴巴，但是模型没这个能力，就用一个居中的distribution来模拟这两个distribution。  
> 关于gradient estimator，是因为对于离散的数据，其gradients的方差会很大。

[13-16]就是先使用ML预训练模型，然后在此基础上adversarial fine-tune.[17-18]则说明了 “that the best-performing GANs tend to stay close to the solution given by maximum-likelihood training”.

所以作者为了证明language GAN真的能work，就from scratch训练了一个language GAN, 对，没有预训练。作者认为从头训练好language GAN的核心技术是 **large batch sizes, dense rewards and discriminator regularization**.

本文的贡献：  
1. 从头训练一个language GAN能达到基于ML方法的unconditional text generation.  
2. 证明 **large batch sizes, dense rewards and discriminator regularization** 对于训练language GAN的重要性。  
3. 作者对文本生成模型的evaluation提出了一些性的拓展，能充分挖掘生成的language更多的特性。比如：
    - BLEU and Self-BLEU [19] capture basic local consistency.    
    - The Frechet Distance metric [17] captures global consistency and semantic information.    
    - Language and Reverse Language model scores [18] across various softmax temperatures to capture the diversity-quality trade-off.    
    - Nearest neighbor analysis in embedding and data space provide evidence that our model is not trivially overfitting.   

### Generative Models of Text
生成模型的本质就是对unknown data distribution进行建模，也就是学习模型 p(x|y) 的参数。在传统的机器学习里面，我们认为模型 p(x|y) 的分布就是多维高斯正态分布，然后用EM算法去学习得到参数。在基于DL的自然语言处理领域，$x=[x_1,..,x_T]$ 的序列特性使得其非常适合使用自回归模型进行建模:
$$p_{\theta}=\prod_{t=1}^Tp_{\theta}(x_t|x_1,...,x_{t-1})$$

### Maximum Likelihood
一旦模型建立好了，接下来就是训练模型。最常用的方法就是使用极大似然估计 maximum likelihood estimation(MLE).

$$\argmax_{\theta}\mathbb{E}_{p^* (x)}logp_{\theta}(x)$$

关于 maximum likelihood 是否是最优解，这篇paper有讨论[9]。

### Generative Adversarial Networks
![](从0开始GAN-4-ScratchGAN/gans.png)
前面seqgan也说过自回归模型中 $p_{\theta}=\prod_{t=1}^Tp_{\theta}(x_t|x_1,...,x_{t-1})$的过程有个sample的操作，这是不可导的。针对这个问题，有三种解决方法：  
- 高方差，无偏估计的 reinforce[28]. 基于大数定律的条件下，去sample更多的example，然后基于policy gradient去学习对应的distribution，这使得速度很慢。  
- 低方差，有偏估计的 gumbel-softmax trick[29-30].  
- other continuous relaxations[11].  

### Learning Signals
对于generator的训练，作者采用了基于 REINFORCE[28] 的方法:
![](从0开始GAN-4-ScratchGAN/reinforce.png)







reference:
[9] Ferenc Husz´ar. How (not) to train your generative model: Scheduled sampling, likelihood, adversary? arXiv
[12-16]
[17-18]
[28]
