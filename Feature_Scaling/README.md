## Feature scaling
***
* Feature scaling is a method used to normalize the range of independent variables or features of data. 
* In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
* Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 
* For example, many classifiers calculate the distance between two points by the Euclidean distance. 
* If one of the features has a broad range of values, the distance will be governed by this particular feature. 
* Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it

## Before Scaling
1. Orginal Data Set Distribution![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/UnScaling%20Data.svg)
## After Scaling
1. Standard Scaler ![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/Standard%20Scaler%20Data.svg)
2. Min_Max_scaler ![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/MinMaxScaler%20Data.svg)
3. Robust Scaler ![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/RobustScaler%20Data.svg)
4. MaxAbsScaler![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/MinMaxScaler%20Data.svg)
5. PowerTransformer(Box Cox)![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/Power%20Transofmation%20Box%20Cox%20transforms.svg)
6. PowerTransformer(Yeo-Johnson)![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/Yeo-Johnson.svg)
7. Quantile_transform_uniform![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/QuantileTransformer%20(uniform).svg)
8. Quantile_transform_Gaussian![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/QuantileTransformer%20(uniform).svg)
9. Normalizer![](https://github.com/reddyprasade/Machine-Learning-with-Scikit-Learn-Python-3.x/blob/master/img/Feature%20Scaling/Normilizer.svg)


### Reference 
1. [Wiki](https://en.wikipedia.org/wiki/Feature_scaling)

## On What Data We Have to apply What  type of Scaling:
![](https://lh3.googleusercontent.com/-JgXpB4x3V0c/Xqa1jaEze1I/AAAAAAAAn5E/IF970DPGHWcMieoywvWyMAzj19drP1ywACK8BGAsYHg/s0/25.jpg)
