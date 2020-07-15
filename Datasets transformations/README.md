# Preprocessing data
The `sklearn.preprocessing` package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.

* In general, learning algorithms benefit from standardization of the data set. 
* If some outliers are present in the set, robust scalers or transformers are more appropriate. 
* The behaviors of the different scalers, transformers, and normalizers on a dataset containing marginal outliers is highlighted in Compare the effect of different scalers on data with outliers.
