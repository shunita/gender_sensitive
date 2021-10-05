# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:40:27 2020

@author: shunit.agmon
"""
import numpy as np
from scipy.stats import norm

def AUROC_confidence_interval(AUC, Npos, Nneg, alpha=0.05):
    # Calculated according to Hanley & McNeil (1982)
    # For large samples, AUC is from a normal distribution
    # N1, N2 are the sizes of positive (N1) and negative (N2) examples
    N1, N2 = Npos, Nneg
    q0 = AUC*(1-AUC)
    q1 = AUC/(2-AUC) - AUC**2
    q2 = 2*(AUC**2)/(1 + AUC) - AUC**2
    se = np.sqrt((q0 + (N1-1)*q1 + (N2-1)*q2)/(N1*N2))
    z_crit = norm.ppf(1-alpha/2)
    lower = max(0, AUC - z_crit*se)
    upper = min(1, AUC + z_crit*se)
    return lower, upper