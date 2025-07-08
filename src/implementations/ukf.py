import numpy as np
from scipy.linalg import cholesky

class UnscentedKalmanFilter:
    def __init__(self, f, h, Q, R, P, x0, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        Initialize the Unscented Kalman Filter.

        Parameters:
        - f: Nonlinear process model function: x_k+1 = f(x_k, u_k)
        - h: Nonlinear measurement model function: z_k = h(x_k)
        - Q: Process noise covariance matrix
        - R: Measurement noise covariance matrix
        - P: Initial state covariance matrix
        - x0: Initial state estimate
        - alpha, beta, kappa: Parameters that control spread of sigma points
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0
        self.n = len(x0)  # State dimension
        self.I = np.eye(self.n)

        # Unscented transform parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)  # Scaling factor

        # Compute weights for sigma points
        self.Wm = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambda_))  # weights for mean
        self.Wc = self.Wm.copy()  # weights for covariance
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    def _generate_sigma_points(self, x, P):
        """
        Generate 2n+1 sigma points from state mean x and covariance P.

        The sigma points are deterministically chosen to match the mean and
        covariance of the current estimate.
        """
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x  # First sigma point is the mean

        # Compute square root of scaled covariance matrix using Cholesky
        try:
            sqrt_P = cholesky(P)
        except np.linalg.LinAlgError:
            sqrt_P = cholesky(P + 1e-6 * np.eye(self.n))  # numerical safety

        for i in range(self.n):
            sigma_points[i + 1] = x + self.gamma * sqrt_P[i]
            sigma_points[self.n + i + 1] = x - self.gamma * sqrt_P[i]

        return sigma_points

    def predict(self, u=None):
        """
        Prediction step of the UKF.

        - Propagates sigma points through the process model
        - Computes predicted state mean and covariance
        """
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # Propagate each sigma point through the process model
        X_pred = np.array([self.f(sp, u) if u is not None else self.f(sp) for sp in sigma_points])

        # Compute predicted mean from propagated sigma points
        self.x = np.sum(self.Wm[:, None] * X_pred, axis=0)

        # Compute predicted covariance
        dx = X_pred - self.x
        self.P = np.sum([self.Wc[i] * np.outer(dx[i], dx[i]) for i in range(2 * self.n + 1)], axis=0) + self.Q
        self.P = (self.P + self.P.T) / 2  # enforce symmetry

        # Store predicted sigma points for use in update
        self._X_pred = X_pred
        return self.x.copy()

    def update(self, z):
        """
        Update step of the UKF.

        - Transforms predicted sigma points into measurement space
        - Computes innovation and Kalman gain
        - Updates state and covariance estimate
        """
        # Transform predicted sigma points through measurement function
        Z_pred = np.array([self.h(sp) for sp in self._X_pred])

        # Compute predicted measurement mean
        z_mean = np.sum(self.Wm[:, None] * Z_pred, axis=0)

        # Measurement residuals
        dz = Z_pred - z_mean
        dx = self._X_pred - self.x

        # Innovation covariance
        S = np.sum([self.Wc[i] * np.outer(dz[i], dz[i]) for i in range(2 * self.n + 1)], axis=0) + self.R
        S = (S + S.T) / 2  # enforce symmetry

        # Cross covariance between state and measurement
        Pxz = np.sum([self.Wc[i] * np.outer(dx[i], dz[i]) for i in range(2 * self.n + 1)], axis=0)

        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(S)

        # Update state estimate and covariance
        y = z - z_mean  # measurement residual (innovation)
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.P = (self.P + self.P.T) / 2  # enforce symmetry

        return self.x.copy()

    def get_state(self):
        """Return the current state estimate."""
        return self.x.copy()

    def get_covariance(self):
        """Return the current estimate covariance."""
        return self.P.copy()
