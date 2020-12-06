# imports
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # dimension reduction & scaling
        lsvc = LinearSVC(C = 0.002, penalty="l1", dual=False, max_iter = 500)

        preprocc_pipe = Pipeline([
                            ('Scaler', StandardScaler()),
                            ('SelectFrom', SelectFromModel(lsvc, prefit = False))
                        ])

        # Add model
        model_pipe = Pipeline([
                        ('preprocessing', preprocc_pipe),
                        ('model_SVM', SVC(kernel = 'rbf', C = 2.11111, gamma = 0.00833))
                    ])

        return model_pipe


    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on test set"""
        acc = self.pipeline.score(X_test, y_test)

        return acc

    def class_report(self, X_test, y_test):
        """return the classification report"""

        print(classification_report(y_test, self.pipeline.predict(X_test)))


class Prepoccess():

    def __init__(self):
        return None

    def feature_selec(self, C = 0.002):
        # Using linear SVC as feature selection
        lsvc = LinearSVC(C = C, penalty="l1", dual=False, max_iter = 500)
        selection_model = SelectFromModel(lsvc, prefit = False)

        return selection_model






class Models():

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def model_SVM(self, ker = 'rbf', C = 2.11111, gamma = 0.00833):
        svm = SVC(kernel = ker, C = C, gamma = gamma)

        return svm


    def model_tree(self, max_depth = 8, min_samples_leaf: 0.001, min_samples_split : 0.003):
        cl_tree = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split)

        return cl_tree


    def model_rand_forest(self):
        cl_forest = RandomForestClassifier(n_estimators = 250, max_depth = 20, min_samples_leaf = 0.001)

        return cl_forest


    def model_xgb(self)
        params = {}
        params['learning_rate'] = 0.2          # 0.01 - 0.2
        params['n_estimators'] = 165
        params['subsample'] = 0.65              # Fraction of observations to be use
        params['colsample_bytree'] = 0.75       # Fraction of features to be use
        params['max_depth'] = 5                # 5/15

        xgb_cl = XGBClassifier(objective = 'multi:softmax', **params)

        return xgb_cl














