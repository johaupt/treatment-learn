"""
Evaluation of treatment effect estimation
"""

import numpy as np

def transformed_outcome_loss(tau_pred, y_true, g, prob_treatment):
    """
    Calculate a biased estimate of the mean squared error of individualized treatment effects

    tau_pred : array
      The predicted individualized treatment effects.
    y_true : array
      The observed individual outcome.
    g : array, {0,1}
      An indicator of the treatment group. Currently supports only two treatment groups, typically
      control (g=0) and treatment group (g=1).
    """
    # Transformed outcome
    y_trans = (g - prob_treatment)  * y_true / (prob_treatment * (1-prob_treatment))
    loss = np.mean(((y_trans - tau_pred)**2))
    return loss


def qini_score(tau_score, y_true, treatment_group, prob_treatment, n_bins=10):
    """
    Calculate the Qini score
    """
    return NotImplementedError


def expected_policy_profit(targeting_decision, g, observed_profit, prob_treatment):
    """
    Calculate the profit of a coupon targeting campaign
    """
    return np.sum(((1-targeting_decision) * (1-g) * observed_profit)/(1-prob_treatment) +\
                   (targeting_decision  *    g  * observed_profit)/(prob_treatment))
