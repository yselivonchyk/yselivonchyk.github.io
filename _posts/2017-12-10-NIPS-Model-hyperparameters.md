NIPS2017 is over. It has been 6 intense days full of philosophy, research, science, inspiration and idea exchange. It is time to organize the knowledge now. Give it some structure and document some outtakes while they are still fresh.

I would try to partition the narration into next uneven sections:
* Trends and research direction. Problems yet to be solved.
* Scientific progress. Problems being solved and solved again.
* Corporate interest. See what influential industry players are trying to achieve and how far they got.
* Philosophical outtakes. The guiding lights in the wast world of neural processing.

That said, lets jump right in.


# Scientific progress: selecting training parameters

## Background. Hyper-parameter

After spending 2 month trying to improve text classification architecture by [Yoon Kim][1] it was strikingly clear to me: hyper-parameter search for a novel architecture is long and convoluted process and require lots of insight and experimentation. And, I am afraid, Deep learning book is a little when your training consistently returns NaN as cross-entropy loss for the most settings configurations.

Model hyper-parameters have complex interrelation. For instance, it is commonly known, that increasing batch size (BS) allows increasing learning rate (LR). This increase can be proportional at first but for significantly larger batch sizes starts to skew towards logarithmic. Which means diminishing returns of increased batch size and non-linear behavior between 2 most common parameters. In addition, this non-linear dependence also means that training has to run longer untill convergence for larger batch sizes. It brings TOTAL_EPOCHS parameter into play (unless you have a great rule for the task at hand for stopping the training automatically).

Unfortunately there are more dimensions to it. First, some per-batch operations have to be adjusted. Weight decay being computed for every batch has to be scaled. According to which law? Well, one might typically just keep it as high as possible as long as training is stable. And, from my experience, there is no intuitive rule for figuring out the sweet spot for weight decay, it is just some seemingly random point. Past that point your training would rapidly fall apart.

Another player is Batch Normalization (BN). And yes, you can not influence it at all: it is either on or off. Yet, it is useful to understand what effects might it have. So, let me scaffold properties of BN first.

## Background. Batch Normalization.

[Batch normalization][3] is one of the least understood normalization techniques. While the original publication is brilliant and outlines all necessary information it often takes some practice to get comfortable with it. I will try to summarize BN effects for our task of better understanding hyper-parameter inter-relations:

1. stabilizing the training. BN decreases variance of neuron outputs between batches. It reduces training noise by simplifying subsequent layer weight co-adaptation.
2. injecting noise. For a given layer *N* and example *i* from the batch, activations *f(x_i)* of the layer would be conditionally dependent on the other examples in the batch. This represents noise i.e. for fixed set of network parameters single input produces different activations depending on the batch.
3. model generalization. Ruffly speaking, BN not only insures consistent distribution of neuron activations in between batches but also in between training set and inference data. These 2 distributions are supposed to be the same, which might or might not be true for real world scenerio. Additionally, we don't have access to the distribution itself and have to work with often nasty samples.

What happens to these properties when the batch size increases?

First, training becomes more stable (property 1). Since BN uses per-batch first and second order moments as approximations of actual (full dataset) moments, increasing batch size yields better approximations. Better moment approximation leads to lower noise during training and, therefore,  more stable training. You might try to think about BN effect for BS=1 to have a better idea of why larger BS lead to more stable training.

Second, the noisy effect on neuron activation would decrease (property 2). The argument is the same as in previous paragraph. Better moment approximations would yield input activations with less noise in them. Subsequently, regularization effect would decrease. Here one might want to think about increasing the other regularizers (weight decay, for example).

Clearly, navigating in hyper-parameter space takes a lot of practice. As I showed above altogether we have 5 entangled hyper-parameters present in almost every NN model: learning rate, batch size, batch normalization, weight decay and number of training epochs. And this is not an exhaustive list. The better performance you try to squeeze out of your model the more often you encounter another unexpected relation between things you did not see as connected.

Yet, it is very tempting to navigate in this highly counterintuitive space. While a good scientific publication would avoid messing with hyper-parameters altogether to minimize amount of unnecessary information and improve reproducibility, industry has other urge: increasing batch size. Larger batch sizes allow us to utilize the computing power better, by using more memory or using memory more effectively. Larger batch sizes naturally emerge from effective distributed learning. Network (GPU?) communication overhead is typically almost constant per-batch, regardless of the size.

That leads us to the this year NIPS publications. They view the problem from different perspective and describe a few techniques and experimental results. In short, I would say, that getting familiar with these publications helps to build a strong understanding of how to work with these hyper-parameters subset.

## NIPS2017 contributions


NIPS2016 shok the deep learning society a bit with the bold claim. Presumably, [large-batch methods tend to converge to sharp minima][2]. Which, of cause, was not taken lightly and resulted in a number of amazing publications this year. It was a reappearing topic showing how this statement is not true in many ways. One of the most exciting developments supported by impressive (and costly) visualizations can be found [here][7]. Author shows how "sharpness" of the minima depends on the weight norm. It also provides some impressive pictures of the loss landscape along random coordinates of the loss space. On the picture below you can see 2 loss surfaces corresponding to training with and without batch normalization.

In [next paper][6] authors argue, that learning rate can be scaled with the BS as long as weight updates are smaller than the weights themselves. It intuitively tells us to stop increasing LR at some point. But, authors propose per-layer LR selection, which allows faster training. As the result, they were able to train ResNet-50 on ImageNet dataset with batch size 8000 and over. The limiting factor is that it does not necessarily work for newer architectures.

Another view on the issue is the training schedule. GoogleAI researches make it clear already in the name of the article [Don't Decay the Learning Rate, Increase the Batch Size][4].

A [new paper][5] by Sergey Ioffe, the author of original [batch normalization][3] technique, shows how to further improve generalization properties of the models using BN. The paper looks deeper into differences of applying BN at train and inference times and shares some useful practical insights. The new BN would be particulaly usefull for small batch sizes and for not i.i.d. batches. The implementation is probably already available in your deep learning framework of choice.

[1]: https://arxiv.org/abs/1408.5882
[2]: https://arxiv.org/abs/1609.04836
[3]: https://arxiv.org/abs/1502.03167
[4]: https://arxiv.org/abs/1711.00489
[5]: https://arxiv.org/abs/1702.03275
[6]: https://arxiv.org/abs/1708.03888
[7]: https://openreview.net/forum?id=HkmaTz-0W
