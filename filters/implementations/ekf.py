import numpy as np
from scipy.linalg import sqrtm


class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, P, x0):
        """
        Initialize the Extended Kalman Filter.

        Parameters:
        f - Function for state transition: f(x, u)
        h - Function for measurement prediction: h(x)
        F_jacobian - Function returning Jacobian of f with respect to x
        H_jacobian - Function returning Jacobian of h with respect to x
        Q - Process noise covariance
        R - Measurement noise covariance
        P - Initial estimate error covariance
        x0 - Initial state estimate
        """
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0
        self.I = np.eye(len(x0))