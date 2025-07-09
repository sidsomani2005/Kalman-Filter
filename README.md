# Kalman-Filter
Comparative experiments investigating the Kalman Filter and performance compared

<br>

# Implementations Included

### 1. Linear Models
- g-h Filter
- Univariate (1D) Kalman Filter
- Multivariate Kalman Filter

### 2. Non-linear Models
- Extended Kalman Filter
- Uuscented Kalman Filter

<br>

## Extended Kalman Filter (EKF)
### - Parameters
        - f: Function for state transition: f(x, u)
        - h: Function for measurement prediction: h(x)
        - F_jacobian: Function returning Jacobian of f with respect to x
        - H_jacobian: Function returning Jacobian of h with respect to x
        - Q: Process noise covariance
        - R: Measurement noise covariance
        - P: Initial estimate error covariance
        - x0: Initial state estimate

### - State Prediction
- Linearize process model:  
  $F = \frac{\partial f}{\partial x}$
- Predict state:  
  $x' = f(x, u)$
- Predict covariance:  
  $P' = F P F^\top + Q$

### - Updating State
- Linearize measurement model:  
  $H = \frac{\partial h}{\partial x}$
- Measurement prediction:  
  $\hat{z} = h(x')$
- Innovation:  
  $y = z - \hat{z}$
- Innovation covariance:  
  $S = H P' H^\top + R$
- Kalman gain:  
  $K = P' H^\top S^{-1}$
- Update state:  
  $x = x' + K y$
- Update covariance (Joseph form):  
  $P = (I - K H) P' (I - K H)^\top + K R K^\top$

<br>

## Unscented Kalman Filter (UKF)
### - Parameters
        - f: Nonlinear process model function: x_k+1 = f(x_k, u_k)
        - h: Nonlinear measurement model function: z_k = h(x_k)
        - Q: Process noise covariance matrix
        - R: Measurement noise covariance matrix
        - P: Initial state covariance matrix
        - x0: Initial state estimate
        - alpha, beta, kappa: Parameters that control spread of sigma points

### - State Prediction
- Generate sigma points from current state and covariance.
- Propagate each sigma point through process model:  
  $X_i' = f(X_i, u)$
- Predicted mean:  
  $x' = \sum W_m^{(i)} X_i'$
- Predicted covariance:  
  $P' = \sum W_c^{(i)} (X_i' - x')(X_i' - x')^\top + Q$

### - Updating State
- Transform predicted sigma points through measurement function:  
  $Z_i = h(X_i')$
- Predicted measurement:  
  $\hat{z} = \sum W_m^{(i)} Z_i$
- Innovation:  
  $y = z - \hat{z}$
- Innovation covariance:  
  $S = \sum W_c^{(i)} (Z_i - \hat{z})(Z_i - \hat{z})^\top + R$
- Cross-covariance:  
  $P_{xz} = \sum W_c^{(i)} (X_i' - x')(Z_i - \hat{z})^\top$
- Kalman gain:  
  $K = P_{xz} S^{-1}$
- Update state:  
  $x = x' + K y$
- Update covariance:  
  $P = P' - K S K^\top$


... Basic experiments completed -> in depth optimizations to come!

