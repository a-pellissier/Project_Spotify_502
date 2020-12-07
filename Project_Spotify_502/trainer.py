# imports
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


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
        lsvc = LinearSVC(C = 0.0008, penalty="l1", dual=False, max_iter = 500)

        preprocc_pipe = Pipeline([
                            ('Scaler', StandardScaler()),
                            ('SelectFrom', SelectFromModel(lsvc, prefit = False))
                        ])

        # Add model
        model_pipe = Pipeline([
                        ('preprocessing', preprocc_pipe),
                        ('model_SVM', SVC(kernel = 'rbf', C = 2.11111))
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

    def conf_matrix(self, X, y):
        y_pred = self.pipeline.predict(X)

        class_labels = self.y.astype('str').unique().tolist()

        cf_mat = confusion_matrix(y, y_pred, labels = class_labels)


        cmtx = pd.DataFrame(
            cf_mat,
            index=['true:{:}'.format(x) for x in class_labels],
            columns=['pred:{:}'.format(x) for x in class_labels])

        return cmtx



class Preprocess():

    def __init__(self, X, Y):
        self.X = X
        self.y = y
        self.pipe_prepro = None


    def feature_selec(self, C = 0.0008):
        # Using linear SVC as feature selection
        lsvc = LinearSVC(C = C, penalty="l1", dual=False, max_iter = 500)
        selec_from_model = SelectFromModel(lsvc, prefit = False)

        return selec_from_model


    def classes_to_undersamp(self, ratio_to_minority = 0.25):
        df_count = pd.DataFrame(self.y.rename('count').value_counts())

        min_val = df_count['count'].min()

        df_count = df_count[df_count['count'] > min_val / ratio_to_minority]

        cat_undersample = {g : int(min_val / ratio_to_minority) for g in df_count.index}

        return cat_undersample


    def resample(self, under_strat = 'auto', over_strat = 'auto'):
        # under_sampling
        undersamp = RandomUnderSampler(sampling_strategy = under_strat)

        # over_sampling
        oversamp = SMOTE(sampling_strategy = over_strat)

        pipe_sample = Pipeline([
            ('under_sampling', undersamp),
            ('over_sampling', oversamp)
            ])

        return pipe_sample

"""
    def set_prepropipe(self, feature_selection = True, resampling = False, C_feat_sel = 0.002, ):
        steps = [('Scaler', StandardScaler())]

        if feature_selection == True:
            selection_model = feature_selec(C = C_feat_sel)
            steps.append((''))
"""



class Models():

    def __init__(self):
        return None


    def model_SVM(self, ker = 'rbf', C = 2.11111):
        svm = SVC(kernel = ker, C = C)

        return svm


    def model_tree(self, max_dep = 8, min_samp_lf = 0.001, min_samp_splt = 0.003):
        cl_tree = DecisionTreeClassifier(max_depth = max_dep, min_samples_leaf = min_samp_lf, min_samples_split = min_samples_splt)

        return cl_tree


    def model_rand_forest(self):
        cl_forest = RandomForestClassifier(n_estimators = 250, max_depth = 20, min_samples_leaf = 0.001)

        return cl_forest


    def model_xgb(self):
        params = {}
        params['learning_rate'] = 0.1          # 0.01 - 0.2
        params['n_estimators'] = 180
        params['subsample'] = 0.8              # Fraction of observations to be use
        params['colsample_bytree'] = 0.8       # Fraction of features to be use
        params['max_depth'] = 10               # 5/15

        xgb_cl = XGBClassifier(objective = 'multi:softmax', **params)

        return xgb_cl














