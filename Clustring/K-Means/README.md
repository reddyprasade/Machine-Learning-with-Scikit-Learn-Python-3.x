k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. 



# Difference Between Kmeans annd KMean++

K-means starts with allocating cluster centers randomly and then looks for "better" solutions. K-means++ starts with allocation one cluster center randomly and then searches for other centers given the first one. So both algorithms use random initialization as a starting point, so can give different results on different runs. As an example you can check this lecture: Clustering As An Example Inference Problem, around 40th minute there are examples of k-means runs, but whole lecture is interesting.

So, answering your questions:

* No, because there is a random initialization different runs can give different results (see examples in the lecture). They should give comparable results but this is not guaranteed. Also, as all the centers are initialized randomly in k-means, it can give different results than k-means++.
* K-means can give different results on different runs.
* The k-means++ paper provides monte-carlo simulation results that show that k-means++ is both faster and provides a better performance, so there is no guarantee, but it may be better.
As about your problem: what k-means++ does it chooses the centers and then starts a "classic" k-means. So what you can do is (1) use the part of algorithm that chooses centers and then (2) use those centers in the GPU implementations of k-means. This way at least a part of a problem is solved on GPU-based software, so should be faster.
