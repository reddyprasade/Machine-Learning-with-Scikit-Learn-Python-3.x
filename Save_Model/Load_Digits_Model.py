import joblib
from sklearn.datasets import load_digits
filename = 'digits_classifier.joblib.pkl'

digits = load_digits()

clf2 = joblib.load(filename)
print(clf2.score(digits.data, digits.target))
