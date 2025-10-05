"""
Endpoint CFM: Conditional Flow Matching for Endpoint Prediction

A package for training and using conditional flow matching models 
for predicting final states in dynamical systems.
"""

from .orchestration import EndpointCFM

__version__ = "0.1.0"
__all__ = ["EndpointCFM"]