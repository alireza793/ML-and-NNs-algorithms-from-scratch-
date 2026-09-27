import numpy as np

def leaky_relu(x, alpha=0.01):
    """تابع فعال‌سازی Leaky ReLU."""
    return np.where(x > 0, x, alpha * x)

# تست تابع
x = np.array([-2, -1, 0, 1, 2])
output = leaky_relu(x)
print(output)
