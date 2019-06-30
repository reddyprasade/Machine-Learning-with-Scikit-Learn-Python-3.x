import numpy as np
from sklearn.linear_model import LinearRegression


# Dats Set
x = np.array([[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]])
y = np.array([4, 5, 20, 14, 32, 22, 38, 43])

#Model Selection
model = LinearRegression().fit(x, y)

#Get results 
r_sq = model.score(x, y)
print('coefficient of determination:\n', r_sq)
print('intercept:\n', model.intercept_)
print('slope:\n', model.coef_)

# Predict response
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')


y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print('predicted response:', y_pred, sep='\n')


# You can apply this model to new data as well
x_new = np.arange(10).reshape((-1, 2))
print(x_new)

# New Predications
y_new = model.predict(x_new)
print(y_new)
