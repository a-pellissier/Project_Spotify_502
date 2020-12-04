# imports
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report

import numpy as np


class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self, C_param = 0.002):
        """defines the pipeline as a class attribute"""

        # dimension reduction & scaling
        lsvc = LinearSVC(C = C_param, penalty="l1", dual=False, max_iter = 500)

        preprocc_pipe = Pipeline([
                            ('Scaler', StandardScaler()),
                            ('SelectFrom', SelectFromModel(lsvc, prefit = False))
                        ])

        # Add model
        model_pipe = Pipeline([
                        ('preprocessing', preprocc_pipe),
                        ('SVM', SVC())
                    ])

        return model_pipe


    def run(self, C_param = 0.002):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline(C_param)
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on test set"""
        acc = self.pipeline.score(X_test, y_test)

        return acc

    def class_report(self, X_test, y_test):
        """return the classification report"""

        print(classification_report(y_test, self.pipeline.predict(X_test)))
