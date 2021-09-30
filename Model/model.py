from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

# Define feature matrix in "X"
X = iris.data

# Define target response vector in "y"
y = iris.target

X_df = pd.DataFrame(data=X, columns = iris.feature_names)
X_df.describe()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

clf_rf = RandomForestClassifier(random_state = 1, n_estimators = 10, n_jobs = -1)
estimator_rf = clf_rf
estimator_rf.fit(X=X_train, y=y_train)

estimator_rf.score(X_test,y_test)

estimator_cv = clf_rf
scores = cross_val_score(estimator_cv, X, y, cv = 5, scoring = 'accuracy')
scores.mean()

dump(estimator_rf, '../Prediction/classifier/IRISRandomForestClassifier.joblib') 

