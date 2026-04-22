import numpy as np

L1_CORTICAL_PARCELS = 148
L2_NETWORKS = 50
L3_NETWORKS = 17

def generate_positive_weights(n_subjects=5):
    """
    Generates dummy weight matrices with values between 0 and 1.
    
    Returns:
        weights_L1_to_L2: Array of shape (n_subjects, L1_CORTICAL_PARCELS, L2_NETWORKS)
        weights_L2_to_L3: Array of shape (n_subjects, L2_NETWORKS, L3_NETWORKS)
    """
    # np.random.rand creates a uniform distribution between 0 and 1
    weights_L1_to_L2 = np.random.rand(n_subjects, L1_CORTICAL_PARCELS, L2_NETWORKS)
    weights_L2_to_L3 = np.random.rand(n_subjects, L2_NETWORKS, L3_NETWORKS)
    
    print(f"Generated positive weights for {n_subjects} subjects.")
    return weights_L1_to_L2, weights_L2_to_L3

def compute_L1_to_L3_weights(weights_L1_to_L2, weights_L2_to_L3):
    """
    Compute L1→L3 weights using vectorized matrix multiplication.
    """
    if weights_L1_to_L2.shape[1:] != (L1_CORTICAL_PARCELS, L2_NETWORKS):
        raise ValueError(
            f"weights_L1_to_L2 must have shape (n, {L1_CORTICAL_PARCELS}, {L2_NETWORKS}), "
            f"got {weights_L1_to_L2.shape}"
        )
    
    if weights_L2_to_L3.shape[1:] != (L2_NETWORKS, L3_NETWORKS):
        raise ValueError(
            f"weights_L2_to_L3 must have shape (n, {L2_NETWORKS}, {L3_NETWORKS}), "
            f"got {weights_L2_to_L3.shape}"
        )
    
    if weights_L1_to_L2.ndim != 3:
        raise ValueError(f"weights_L1_to_L2 must be 3D, got {weights_L1_to_L2.ndim}D")
    
    if weights_L1_to_L2.shape[0] != weights_L2_to_L3.shape[0]:
        raise ValueError(
            f"Subject count mismatch: weights_L1_to_L2 has {weights_L1_to_L2.shape[0]} "
            f"subjects, weights_L2_to_L3 has {weights_L2_to_L3.shape[0]}"
        )

    # 1. Batch matrix multiply: (n, 148, 50) @ (n, 50, 17) -> (n, 148, 17)
    weights_L1_to_L3 = weights_L1_to_L2 @ weights_L2_to_L3
    
    # 2. Reshape to flatten the 148x17 matrices for each subject
    # -1 tells NumPy to calculate the size of that dimension automatically
    return weights_L1_to_L3.reshape(weights_L1_to_L3.shape[0], -1)