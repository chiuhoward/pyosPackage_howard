import numpy as np

def generate_positive_weights(n_subjects=5):
    """
    Generates dummy weight matrices with values between 0 and 1.
    
    Returns:
        weights_1to2: Array of shape (n_subjects, 148, 50)
        weights_2to3: Array of shape (n_subjects, 50, 17)
    """
    # np.random.rand creates a uniform distribution between 0 and 1
    weights_1to2 = np.random.rand(n_subjects, 148, 50)
    weights_2to3 = np.random.rand(n_subjects, 50, 17)
    
    print(f"Generated positive weights for {n_subjects} subjects.")
    return weights_1to2, weights_2to3

def compute_l1_to_l3_weights(weights_1to2, weights_2to3):
    """
    Compute L1→L3 weights using vectorized matrix multiplication.
    """
    # 1. Batch matrix multiply: (n, 148, 50) @ (n, 50, 17) -> (n, 148, 17)
    weights_1to3 = weights_1to2 @ weights_2to3
    
    # 2. Reshape to flatten the 148x17 matrices for each subject
    # -1 tells NumPy to calculate the size of that dimension automatically
    return weights_1to3.reshape(weights_1to3.shape[0], -1)