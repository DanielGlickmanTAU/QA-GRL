### learning seem to happen in very few epochs; after a few, sometimes even a single one, it is close to best preformence. at least when running for ~ epochs
https://www.comet.ml/danielglickmantau/dl/5332d04179ce424d954e3591b1900e21


### roberta does not learn on race with 'batch_size': 16, 'learning_rate': 3e-05.
run:https://www.comet.ml/danielglickmantau/dl/d97a4af08650437889a9f9bdf93e763b
loss on train seem completely stuck.
it did manage to learn in previous runs.
Maybe the learning rate is too low?

I will give it a few more epocs than
Trying to disable weight decay and increase the learning rate to 6e-0

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