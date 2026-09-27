import numpy as np

def adam_update(weights, grad, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer update step (pure NumPy, no external libraries)
    
    Parameters:
    -----------
    weights : list or np.ndarray
        Current model parameters (theta_t)
    grad : list or np.ndarray
        Gradients of loss w.r.t weights (nabla L)
    m : list or np.ndarray
        First moment vector (exponential moving average of gradients)
    v : list or np.ndarray
        Second moment vector (exponential moving average of squared gradients)
    t : int
        Iteration counter (starts from 1)
    alpha : float
        Learning rate (step size)
    beta1 : float
        Decay rate for first moment (default: 0.9)
    beta2 : float
        Decay rate for second moment (default: 0.999)
    eps : float
        Small constant to prevent division by zero (default: 1e-8)
    
    Returns:
    --------
    weights_new : list or np.ndarray
        Updated parameters
    m_new : list or np.ndarray
        Updated first moment vector
    v_new : list or np.ndarray
        Updated second moment vector
    """
    
    # Initialize lists for updated values
    m_new = []
    v_new = []
    weights_new = []
    
    # Iterate over each parameter (weight)
    for i in range(len(weights)):
        # 1. Update biased first moment estimate (momentum)
        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        m_i = beta1 * m[i] + (1 - beta1) * grad[i]
        m_new.append(m_i)
        
        # 2. Update biased second moment estimate (RMSprop)
        # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        v_i = beta2 * v[i] + (1 - beta2) * (grad[i] ** 2)
        v_new.append(v_i)
        
        # 3. Compute bias-corrected estimates
        # m_hat = m_t / (1 - beta1^t)
        # v_hat = v_t / (1 - beta2^t)
        m_hat = m_i / (1 - beta1 ** t)
        v_hat = v_i / (1 - beta2 ** t)
        
        # 4. Update weights
        # theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
        w_new = weights[i] - alpha * m_hat / (np.sqrt(v_hat) + eps)
        weights_new.append(w_new)
    
    return weights_new, m_new, v_new
