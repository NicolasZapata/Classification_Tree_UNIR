# UNIR - RIOJA INTERNATIONAL UNIVERSITY - COLOMBIA
# CODE EDITING BY: NICOLAS ZAPATA ALZATE

# Base Libraries
from sklearn import tree
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Graphics
import matplotlib.pyplot as pl

# Proccessing and Modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Importing csv
    df = pd.read_csv("src/Admission_Predict.csv")
    df.dropna(how='all')
    # Define Values and Scaler
    x = df.iloc[:, [0, 1]].values
    y = df.iloc[:, 2].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    sc = StandardScaler()

    # Define Train and Test veriables
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Ajusting the Classification Tree
    classifier = DecisionTreeClassifier(criterion='gini', random_state=123, max_depth=5)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy_score(y_pred, y_test)
    df.info()

    # Ploting the results
    clf = DecisionTreeClassifier(criterion='gini', random_state=123, max_depth=5)
    pl.figure(figsize=(40, 20))
    clf = clf.fit(x_train, y_train)
    plot_tree(clf, filled=True)
    pl.title("Decision Tree Admition")
    pl.show()
    text_representation = tree.export_text(clf)
    print(text_representation)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
