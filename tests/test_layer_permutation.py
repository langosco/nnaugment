# import pytest
# import numpy as np
# from nnaugment.core import permute_linear_layer, permute_conv_layer
# 
# # Mock layers
# LINEAR_LAYER = {
#     "kernel": np.array([[1, 2], [3, 4], [5, 6]]),
#     "bias": np.array([1, 2])
# }
# 
# CONV_LAYER = {
#     "kernel": np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),
#     "bias": np.array([1, 2])
# }
# 
# 
# @pytest.mark.parametrize("mode, permutation, expected_kernel, expected_bias", [
#     ("output", [1, 0], np.array([[2, 1], [4, 3], [6, 5]]), np.array([2, 1])),
#     ("input", [2, 0, 1], np.array([[5, 6], [1, 2], [3, 4]]), np.array([1, 2]))
# ])
# def test_permute_linear_layer(mode, permutation, expected_kernel, expected_bias):
#     permuted = permute_linear_layer(LINEAR_LAYER, permutation, mode)
#     np.testing.assert_array_equal(permuted["kernel"], expected_kernel, err_msg=f"Failed for mode={mode} with permutation={permutation}")
#     np.testing.assert_array_equal(permuted["bias"], expected_bias, err_msg=f"Failed for mode={mode} with permutation={permutation}")
# 
#     # Check for invalid mode
#     with pytest.raises(ValueError, match="Unknown mode"):
#         permute_linear_layer(LINEAR_LAYER, permutation, "unknown")
# 
#     # Check for invalid permutation length
#     with pytest.raises(AssertionError):
#         permute_linear_layer(LINEAR_LAYER, [1], mode)
# 
# @pytest.mark.parametrize("mode, permutation, expected_kernel, expected_bias", [
#     ("output", [1, 0], np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]]), np.array([2, 1])),
#     ("input", [1, 0], np.array([[[5, 6], [7, 8]], [[1, 2], [3, 4]]]), np.array([1, 2]))
# ])
# def test_permute_conv_layer(mode, permutation, expected_kernel, expected_bias):
#     permuted = permute_conv_layer(CONV_LAYER, permutation, mode)
#     np.testing.assert_array_equal(permuted["kernel"], expected_kernel, err_msg=f"Failed for mode={mode} with permutation={permutation}")
#     np.testing.assert_array_equal(permuted["bias"], expected_bias, err_msg=f"Failed for mode={mode} with permutation={permutation}")
# 
#     # Check for invalid mode
#     with pytest.raises(ValueError, match="Unknown mode"):
#         permute_conv_layer(CONV_LAYER, permutation, "unknown")
# 
#     # Check for invalid permutation length
#     with pytest.raises(AssertionError):
#         permute_conv_layer(CONV_LAYER, [1], mode)