import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

class MultivariateKalmanFilter:
    def __init__(self, F, H, Q, R, P, x0):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        F - State transition matrix
        H - Observation matrix
        Q - Process noise covariance
        R - Measurement noise covariance
        P - Initial state covariance
        x0 - Initial state estimate
        """
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # State covariance
        self.x = x0  # State estimate
        
        # Identity matrix (for convenience)
        self.I = np.eye(self.F.shape[0])
        
    def predict(self):
        """Predict the next state and covariance."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z):
        """Update the state estimate with new measurement z."""
        y = z - self.H @ self.x  # Innovation/residual
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        
        return self.x