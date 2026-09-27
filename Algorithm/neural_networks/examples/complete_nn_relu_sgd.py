import numpy as np

# تابع فعال‌سازی ReLU
def relu(x):
    return np.maximum(0, x)

# مشتق ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# تابع فعال‌سازی Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# مشتق Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# ورودی
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# خروجی
y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

# وزن‌ها را به صورت تصادفی مقداردهی اولیه می‌کنیم
np.random.seed(1)
weights1 = np.random.rand(3, 2)
weights2 = np.random.rand(2, 2)
weights3 = np.random.rand(2, 2)

# نرخ یادگیری
learning_rate = 0.1

# تعداد دوره‌های آموزش
epochs = 100

# حلقه آموزش
for i in range(epochs):
    # Forward Propagation
    layer1 = relu(np.dot(X, weights1))
    layer2 = relu(np.dot(layer1, weights2))
    output = sigmoid(np.dot(layer2, weights3))

    # محاسبه خطا
    error = y - output

    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    d_layer2 = d_output.dot(weights3.T) * relu_derivative(layer2)
    d_layer1 = d_layer2.dot(weights2.T) * relu_derivative(layer1)

    # به‌روزرسانی وزن‌ها
    weights3 += layer2.T.dot(d_output) * learning_rate
    weights2 += layer1.T.dot(d_layer2) * learning_rate
    weights1 += X.T.dot(d_layer1) * learning_rate

# چاپ وزن‌های نهایی
print("وزن‌های لایه اول:")
print(weights1)
print("\nوزن‌های لایه دوم:")
print(weights2)
print("\nوزن‌های لایه سوم:")
print(weights3)

# پیش‌بینی با استفاده از داده‌های آموزش‌دیده
predictions = sigmoid(np.dot(relu(np.dot(X, weights1)), weights3))
print("\nپیش‌بینی‌ها:")
print(predictions)
