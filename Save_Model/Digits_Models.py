import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
digits = load_digits()
clf = SGDClassifier().fit(digits.data, digits.target)
clf.score(digits.data, digits.target)  # evaluate training error


filename = 'digits_classifier.joblib.pkl'
_ = joblib.dump(clf, filename, compress=9)
print(_)
