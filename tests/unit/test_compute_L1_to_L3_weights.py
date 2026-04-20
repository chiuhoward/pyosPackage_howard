import numpy as np
from pyospackage_howard.helpers import compute_L1_to_L3_weights

L1_CORTICAL_PARCELS = 148
L2_NETWORKS = 50
L3_NETWORKS = 17

def test_compute_L1_to_L3_weights_with_zeros():
    """Test with zero matrices."""
    weights_L1_to_L2 = np.zeros((2, L1_CORTICAL_PARCELS, L2_NETWORKS))
    weights_L2_to_L3 = np.zeros((2, L2_NETWORKS, L3_NETWORKS))
    
    result = compute_L1_to_L3_weights(weights_L1_to_L2, weights_L2_to_L3)
    
    assert np.allclose(result, 0)
    assert result.shape == (2, 2516)