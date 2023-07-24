import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target)

model = RandomForestClassifier()
model.fit(X_train,y_train)

score = model.score(X_test,y_test)
print(score)

# LIME
from lime import lime_tabular
import random

explainer = lime_tabular.LimeTabularExplainer(X_train, mode="classification", feature_names= cancer.feature_names)

idx = random.randint(1, len(X_test))

print("Prediction : ", model.predict(X_test[idx].reshape(1,-1)))
print("Actual :     ", y_test[idx])

explanation = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=len(cancer.feature_names), labels=(0,) ,num_samples=5000)
# explainer.show_in_notebook