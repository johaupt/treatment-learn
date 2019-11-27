"""
Wrapper for indirect estimation of treatment effects using response/outcome models
"""

import numpy as np

class HurdleModel():
    def __init__(self, conversion_classifier, value_regressor):
        self.conversion_classifier = conversion_classifier
        self.value_regressor = value_regressor
    
    def predict(self, X):
        return self.conversion_classifier.predict_proba(X)[:,1] * self.value_regressor.predict(X)  

class TwoModelRegressor():
    def __init__(self, control_group_model, treatment_group_model):
        self.control_group_model = control_group_model
        self.treatment_group_model = treatment_group_model
    
    def predict(self, X):
        return self.treatment_group_model.predict(X) - self.control_group_model.predict(X)

class SingleModel():
    def __init__(self, model, n_treatment=2):
        self.model = model
        self.n_treatment = n_treatment
        if self.n_treatment != 2:
            NotImplementedError("Currently only supports binary treatment")
    
    def predict(self, X):
        return self.model.predict(np.c_[X, np.ones((X.shape[0],1))]) - self.model.predict(np.c_[X, np.zeros((X.shape[0],1))])
    
    def predict_proba(self, X):
        return self.model.predict_proba(np.c_[X, np.ones((X.shape[0],1))]) - self.model.predict_proba(np.c_[X, np.zeros((X.shape[0],1))])

    def fit(self, X, y, treatment_group, **kwargs):
        output = self.model.fit(np.c_[X, treatment_group], y, **kwargs)
        return output
