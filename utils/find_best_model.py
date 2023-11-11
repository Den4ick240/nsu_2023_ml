import numpy as np


def find_best_model(models, x, y, folds=5):
    best_mse = float("inf")
    best_model = None
    fold_size = len(y) // folds
    for model in models:
        mse_sum = 0
        for i in range(folds):
            test_start = i * fold_size
            test_end = test_start + fold_size
            x_test = x[test_start:test_end]
            y_test = y[test_start:test_end]
            x_train = np.concatenate((x[:test_start], x[test_end:]))
            y_train = np.concatenate((y[:test_start], y[test_end:]))

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = np.mean((y_pred - y_test) ** 2)
            mse_sum += mse

        avg_mse = mse_sum / folds

        if avg_mse < best_mse:
            best_model = model
            best_mse = avg_mse

    return best_model
