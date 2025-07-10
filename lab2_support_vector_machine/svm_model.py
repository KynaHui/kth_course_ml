import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize


class SVM:
    def __init__(self, kernel='rbf', C=1e10, poly_exp=3, rbf_sigma=1):
        self.C = C
        self.kernel_type = kernel
        self.kernel = self._get_kernel(kernel, poly_exp, rbf_sigma)
        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.inputs = None
        self.targets = None

    def _get_kernel(self, kernel_type, poly_exp=3, rbf_sigma=1):
        if kernel_type == 'linear':
            return lambda x, y: np.dot(x, y)
        elif kernel_type == 'poly':
            return lambda x, y: (np.dot(x, y) + 1) ** poly_exp
        elif kernel_type == 'rbf':
            return lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / (2 * rbf_sigma ** 2))
        else:
            raise ValueError()

    def fit(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        N = inputs.shape[0]

        # calculate P_ij = t_i * t_j * K
        P = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                P[i, j] = targets[i] * targets[j] * self.kernel(inputs[i], inputs[j])

        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

        def zerofun(alpha):
            return np.dot(alpha, targets)

        alpha0 = np.zeros(N)
        bounds = [(0, self.C) for _ in range(N)]
        constraints = {'type': 'eq', 'fun': zerofun}

        result = minimize(objective, alpha0, bounds=bounds, constraints=constraints)
        self.alpha = result['x']

        threshold = 1e-5
        sv = self.alpha > threshold
        sv_indices = np.where(sv)[0]
        self.support_vectors = inputs[sv]
        self.support_vector_labels = targets[sv]
        self.support_vector_alphas = self.alpha[sv]


        b_sum = 0
        for i in sv_indices:
            kernel_vals = np.array([self.kernel(inputs[i], inputs[j]) for j in range(N)])
            b_sum += (np.sum(self.alpha * targets * kernel_vals) - targets[i])
        self.b = b_sum / len(sv_indices)


    def indicator(self, s):
        result = 0
        for i in range(len(self.support_vectors)):
            result += self.support_vector_alphas[i] * self.support_vector_labels[i] * \
                      self.kernel(s, self.support_vectors[i])
        return result - self.b

    def predict(self, X):
        if X.ndim == 1:
            return np.sign(self.indicator(X))
        else:
            return np.array([np.sign(self.indicator(x)) for x in X])

    def plot_decision_boundary(self, resolution=500):
        x_min, x_max = self.inputs[:, 0].min() - 1, self.inputs[:, 0].max() + 1
        y_min, y_max = self.inputs[:, 1].min() - 1, self.inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.indicator(point) for point in grid])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                    linestyles=['dashed', 'solid', 'dashed'], linewidths=[1, 2, 1])
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap='coolwarm', alpha=0.3)

        pos = self.inputs[self.targets == 1]
        neg = self.inputs[self.targets == -1]
        plt.scatter(pos[:, 0], pos[:, 1], color='blue', marker='o', label='Positive (+1)')
        plt.scatter(neg[:, 0], neg[:, 1], color='red', marker='x', label='Negative (-1)')

        plt.scatter(self.support_vectors[:, 0], self.support_vectors[:, 1], s=100, facecolors='none',
                    edgecolors='green', label='Support Vectors')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('SVM decision boundary')
        plt.legend()
        plt.show()


