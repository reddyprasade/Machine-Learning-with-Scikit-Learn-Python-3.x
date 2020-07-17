### Bagging
* Bagging = Bootstrap + aggregating
* It uses bootstrap resampling to generate L different training sets from the original training set
* On the L training sets it trains L base learners
* During testing it aggregates the L learners by taking their average (using uniform weights for each classifiers), or by majority voting
* The diversity or complementarity of the base learners is not controlled in any way, it is left to chance and to the instability of the base learning method
* The ensemble model is almost always better than the unique base learners if the base learners are unstable (which means that a small change in the training dataset may cause a large change in the result of the training)
* Suppose we have a training set with n samples
* We would like to create L different training sets from this
* Bootstrap resampling takes random samples from the original set with replacement
* Randomness is required to obtain different sets for L rounds of resampling
* Allowing replacement is required to be able to create sets of size n from the original data set of size n
* As the L training sets are different, the result of the training over these set will also be more or less different, independent of what kind of training algorithm we use
* Works better with unstable learners (e.g. neural nets, decision trees)
* Not really effective with stable learners (e.g. k-NN, SVM)
