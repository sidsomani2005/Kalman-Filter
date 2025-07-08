import numpy as np
from scipy.linalg import sqrtm


class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, P, x0):
        """
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


    def predict(self, u=None):
        F = self.F_jacobian(self.x, u)
        self.x = self.f(self.x, u) if u is not None else self.f(self.x)
        self.P = F @ self.P @ F.T + self.Q
        self.P = (self.P + self.P.T) / 2 # ensuring symmetry of covariance matrix
        
        return self.x.copy()
    

    def update(self, z):
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        S = (S + S.T) / 2
        
        try:
            # kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
        
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        self.x = self.x + K @ y
        
        # using Joseph form for numerical stability
        I_KH = self.I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        self.P = (self.P + self.P.T) / 2
        
        return self.x.copy()


    def get_state(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()