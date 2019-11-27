from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

class DoubleRobustTransformer(BaseEstimator, TransformerMixin):
    """Double Robust Label Transformation
    
    Double robust transformation calculates an artificial variable
    which is equal to the treatment effect in expectation if either
    the mean regression or propensity score model are correctly specified.

    Parameters
    ----------
    mean_estimator : object, optional (default=None)
        The regression estimator used to estimate the treatment-conditional 
        mean effect. If ``None``, then the base estimator is 
        ``LinearRegression()``
    propensity_estimator : object, optional (default=None)
        The classifier used to estimate the treatment propensity. If ``None``, 
        then the base estimator is ``LogisticRegression()``
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------

    Examples
    --------
    >>> import numpy as np
    >>> N = 2000
    >>> X = np.random.normal(size=[N, 5])
    >>> w = np.random.binomial(1, 1/(1+np.exp(-(X[:,1]))) , size=N) 
    >>> y= np.dot(X, np.random.uniform(-1,1, size=5)) + 0.05 * w
    >>> y[w==1].mean() - y[w==0].mean()
    >>> dr = DoubleRobustTransformer()
    >>> y_dr = dr.fit_transform(X,y,w)
    >>> y_dr.mean()

    References
    ----------
    .. [1] M. C., Knaus, M. Lechner, A. Strittmatter, "Machine Learning Estimation 
           of Heterogeneous Causal Effects: Empirical Monte Carlo Evidence", 2019. 
    """
    
    def __init__(self, mean_estimator=None, propensity_estimator=None, random_state=None):
        """
        
        """
        self.mean_estimator=mean_estimator
        self.propensity_estimator=propensity_estimator
        
    def _validate_mean_estimator(self, default):
        """
        Make and configure a copy of the `mean_estimator_` attribute.
        """
        if self.mean_estimator is not None:
            self.mean_estimator_ = self.mean_estimator
        else:
            self.mean_estimator_ = default
        
        if self.mean_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    
    def _validate_propensity_estimator(self, default):
        """
        Make and configure a copy of the `propensity_estimator_` attribute.
        """
        # TODO: Should also take a true propensity value for experiments I think
        if self.propensity_estimator is not None:
            self.propensity_estimator_ = self.propensity_estimator
        else:
            self.propensity_estimator_ = default
        
        if self.propensity_estimator_ is None:
            raise ValueError("propensity_estimator cannot be None")
              
        
    def fit_transform(self, X, y, w, sample_weight=None):
        """Calculate the double robust transformed outcome
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape = [n_samples]
            The target values (class labels).
        w : array-like of shape = [n_samples]
            The binary treatment indicator (treatment : 1, control: 0)

        Returns
        -------
        Transformed Outcome : float
        """
    
        # TODO: Need checks on the estimator types
        self._validate_mean_estimator(
            default=LinearRegression())

        mean_estimator = {}
        mean_estimator[0] = clone(self.mean_estimator_)
        mean_estimator[1] = clone(self.mean_estimator_)

        # TODO: Allow to set the random state
        #        if random_state is not None:
        #            _set_random_states(estimator, random_state)

        mean_estimator[0].fit(X[w==0,:], y[w==0])
        mean_estimator[1].fit(X[w==1,:], y[w==1])
            
        # Train one propensity model
        self._validate_propensity_estimator(
            default=LogisticRegression())
        
        propensity_estimator = clone(self.propensity_estimator_)
        propensity_estimator.fit(X, w)
            
        # Double-robust transformation
        e_x = np.clip(propensity_estimator.predict_proba(X), a_min=0.025, a_max=0.975)
            
        y_dr = mean_estimator[1].predict(X) - mean_estimator[0].predict(X) +            (w*    (y-mean_estimator[1].predict(X))/e_x[:,1]) -            ((1-w)*(y-mean_estimator[0].predict(X))/e_x[:,0])
        
        return y_dr.flatten()
       

