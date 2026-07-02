import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    train_errors, val_errors = [], []
    for r in train_sizes:
        n = r
        X_train_slice = np.array(X_train[:n])
        phi_train = np.column_stack([X_train_slice.flatten() ** j for j in range(degree + 1)])
        w = np.matmul(np.linalg.pinv(phi_train),y_train[:n])
        y_pre = np.matmul(phi_train, w)
        train_errors.append(np.mean((y_pre-y_train[:n])**2))

        X_val_arr = np.array(X_val)
        phi_val = np.column_stack([X_val_arr.flatten() ** j for j in range(degree + 1)])
        y_pre = np.matmul(phi_val, w)
        val_errors.append(np.mean((y_pre-y_val)**2))

    diagnosis = None
    final_train_error, final_val_error = train_errors[-1], val_errors[-1]
    if final_train_error > bias_threshold:
        diagnosis = "high_bias"
    elif (final_val_error - final_train_error) > variance_threshold:
        diagnosis = "high_variance"
    else:
        diagnosis = "good_fit"
    
    return {
        "train_errors":train_errors,
        "val_errors":val_errors,
        "diagnosis":diagnosis
    }

