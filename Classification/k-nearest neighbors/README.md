### K-Nearest Neighbors Algorithm:

* The k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression.
* In both cases, the input consists of the k closest training examples in the feature space. 
* The output depends on whether k-NN is used for classification or regression:
* In k-NN classification, the output is a class membership. 
* An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). 
* If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
* In k-NN regression, the output is the property value for the object. 
* This value is the average of the values of k nearest neighbors.
* k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation.
* Both for classification and regression, a useful technique can be to assign weights to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones.

**For example**
* A common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.
* The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value (for k-NN regression) is known. 
* This can be thought of as the training set for the algorithm, though no explicit training step is required.

### Steps For KNN:
1. For getting the predicted class, iterate from 1 to total number of training data points
2. Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric since it’s the most popular method. The other metrics that can be used are Chebyshev, cosine, etc.
3. Sort the calculated distances in ascending order based on distance values
4. Get top k rows from the sorted array
5. Get the most frequent class of these rows
5. Return the predicted class

![](https://lh3.googleusercontent.com/-phenbqm5GaQ/XowfnRiSMLI/AAAAAAAAnrM/neGfpYib9_sPn8b0ndaSNg3IrkTSlFLUQCK8BGAsYHg/s0/2020-04-06.png)
