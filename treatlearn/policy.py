"""
Decision-making based on profit/cost setting
"""

#import numpy as np

def bayesian_targeting_policy(tau_pred, contact_cost, offer_accept_prob, offer_cost, value=None):
    """
    Applied the Bayesian optimal decision framework to make a targeting decision.
    The decision to target is made when the expected profit increase from targeting is strictly
    larger than the expected cost of targeting.

    tau_pred : array-like
      Estimated treatment effect for each observations. Typically the effect on the expected profit.
      If tau_pred is the treatment effect on conversion, 'value' needs to be specified.

    contact_cost : float or array-like
      Static cost that realizes independent of outcome

    offer_cost : float or array-like
      Cost that realizes when the offer is accepted

    value : float or array-like, default: None
      Value of the observations in cases where tau_pred is the
      change in acceptance probability (binary outcome ITE)
    """
    if value:
        tau_pred = tau_pred * value

    return (tau_pred > (offer_accept_prob * offer_cost - contact_cost)).astype('int')
