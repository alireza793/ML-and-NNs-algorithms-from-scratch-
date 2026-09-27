import numpy as np

def relu(x):
    return np.maximum(0, x)

def neural_network(inputs, weights, biases):
    """شبکه عصبی با ۲ ورودی، ۲ لایه مخفی (هر لایه ۲ سلول) و ۱ خروجی با ReLU."""
    layer1 = relu(np.dot(inputs, weights[0]) + biases[0])
    layer2 = relu(np.dot(layer1, weights[1]) + biases[1])
    output = relu(np.dot(layer2, weights[2]) + biases[2])
    return output

# مقداردهی اولیه وزن‌ها و بایاس‌ها
np.random.seed(42)
weights = [np.random.rand(2, 2) for _ in range(3)]
biases = [np.random.rand(2) for _ in range(3)]

# داده‌های آموزشی
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# تست شبکه
predictions = neural_network(inputs, weights, biases)
print("پیش‌بینی‌ها:")
print(predictions)
