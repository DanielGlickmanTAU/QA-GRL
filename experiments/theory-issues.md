###did not seem to learn to learn on Race dataset 
the classfier acurraccy got stuck on 0.75(always answer false). Even thought the same code did
work for SWAG dataset. 
I tried overfitting on a small dataset(100 examples) for 20 epochs, but it still did not work.
In the end it turned out it did learn, just not as easily as with swag.
I tried a tiny tiny dataset(5 examples) for 100 epochs, and saw it did overfit.


### Race dataset with 75% negative examples(1 correct option, 3 distractors) did not seem to learn but converge to a constant of always returning 0
When I changed the dataset to having 50% negative examples, it learned much better. it reached 61% accuraccy(50% is random) after a few epochs.
Still not good enough but it does indicate that
1) the network tends to have less bias torwards the constant mod
2) the problem seems hard enough so that the current architecture is simply not enough