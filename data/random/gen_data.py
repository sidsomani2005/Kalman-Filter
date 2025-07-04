import numpy as np
from scipy.linalg import sqrtm

def generate_data(true_states, F, H, Q, R, n_steps):
    """Generate synthetic noisy measurements from true states."""
    measurements = []
    true_process = []
    
    x = true_states[0]
    for _ in range(n_steps):
        # Process noise
        w = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        x = F @ x + w
        true_process.append(x)
        
        # Measurement noise
        v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        z = H @ x + v
        measurements.append(z)
    
    return np.array(true_process), np.array(measurements)