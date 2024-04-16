import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt
import data

X, Y = data.load_data("data/Housing.csv")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = RandomForestRegressor()

clf.fit(X_train, Y_train)

importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print(importances)

plt.figure(figsize=(10, 10))
plt.scatter(X_test["area"], Y_test, label="Actual Price")
plt.scatter(X_test["area"], clf.predict(X_test), label="Predicted Price")
plt.legend()
plt.show()
