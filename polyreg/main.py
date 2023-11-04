import numpy as np
import matplotlib.pyplot as plt


# Зададим функции, которые мы хотим восстановить
def true_function_1(x):
    return np.cos(x)


def true_function_2(x):
    return 5 * x**3 + x**2 + 5


def true_function_3(x):
    return x * np.sin(x)


# Зададим функцию для создания выборки с ошибками измерения
def generate_data(func, x, error_type="uniform", scale=0.1):
    if error_type == "uniform":
        error = np.random.uniform(-scale, scale, len(x))
    elif error_type == "normal":
        error = np.random.normal(0, scale, len(x))
    y = func(x) + error
    return y


def regression(x, y, M, alpha=0):
    X = np.vander(x, M + 1, increasing=True)
    A = X.T @ X + alpha * np.identity(M + 1)
    B = X.T @ y
    theta = np.linalg.inv(A) @ B
    return theta


# Функция для оценки регрессии на новых данных
def predict(x, theta):
    M = len(theta) - 1
    X = np.vander(x, M + 1, increasing=True)
    y_pred = X @ theta
    return y_pred


# Зададим диапазон значений x
x = np.linspace(0, 2 * np.pi, 100)

# Зададим значения параметра M
M_values = range(16)

graph = False
# Генерируем выборку и оцениваем регрессию для каждой функции и каждого значения M
if graph:
    for func, func_name in [
        (true_function_1, "cos(x)"),
        (true_function_2, "5x^3 + x^2 + 5"),
        (true_function_3, "xsin(x)"),
    ]:
        for M in M_values:
            y = generate_data(func, x, error_type="uniform", scale=1)
            theta = regression(x, y, M, 1)
            y_pred = predict(x, theta)

            # Визуализация результатов
            plt.figure()
            plt.plot(x, y, "bo")
            plt.plot(x, func(x), label="True Function")
            plt.plot(x, y_pred, label="Polynomial Regression")
            plt.title(f"{func_name}, M={M}")
            plt.legend()
            plt.show()
            # plt.savefig(f"{func_name}_{M}.png")

alphas = [0, 0.001, 0.01, 0.1, 1, 10]

for func, func_name in [
    (true_function_1, "cos(x)"),
    (true_function_2, "5x^3 + x^2 + 5"),
    (true_function_3, "xsin(x)"),
]:
    best_M = -1
    best_alpha = -1
    best_mse = float("inf")
    for M in M_values:
        for alpha in alphas:
            mse_sum = 0

            y = generate_data(func, x, error_type="uniform", scale=0.1)
            folds = 5
            fold_size = len(y) // 5

            for i in range(folds):  # Повторяем 10 раз для разных выборок
                test_start = i * fold_size
                test_end = test_start + fold_size
                x_test = x[test_start:test_end]
                y_test = y[test_start:test_end]
                x_train = np.concatenate((x[:test_start], x[test_end:]))
                y_train = np.concatenate((y[:test_start], y[test_end:]))

                theta = regression(x_train, y_train, M, alpha)
                y_pred = predict(x_test, theta)

                mse = np.mean((y_pred - y_test) ** 2)
                mse_sum += mse

            avg_mse = mse_sum / 10

            if avg_mse < best_mse:
                best_M = M
                best_alpha = alpha
                best_mse = avg_mse

    print(
        f"For func: {func_name}, Best M: {best_M}, Best Alpha (λ): {best_alpha}, Best Average MSE: {best_mse}"
    )
