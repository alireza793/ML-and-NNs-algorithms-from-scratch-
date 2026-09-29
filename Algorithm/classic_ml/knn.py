import numpy as np


class KNN:
    def __init__(self, k=3, task="classification"):
        self.k = k
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        if self.task == "classification":
            values, counts = np.unique(k_labels, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return np.mean(k_labels)

    def score(self, X, y):
        predictions = self.predict(X)
        if self.task == "classification":
            return np.mean(predictions == y)
        else:
            return 1 - np.sum((predictions - y) ** 2) / np.sum((y - np.mean(y)) ** 2)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for k in [1, 3, 5, 7, 11]:
        knn = KNN(k=k, task="classification")
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        print(f"k={k}: Accuracy = {acc * 100:.2f}%")
