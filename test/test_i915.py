import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

class TestI915(unittest.TestCase):
  def setUp(self):
    # Skip tests if I915 is not enabled
    if not getenv("I915"):
      self.skipTest("I915 backend not enabled, set I915=1 to run these tests")

  def test_simple_add(self):
    # Create tensors on I915 device
    a = Tensor([1, 2, 3, 4], device="I915")
    b = Tensor([5, 6, 7, 8], device="I915")
    
    # Perform a simple addition
    c = a + b
    
    # Check the result
    expected = np.array([6, 8, 10, 12])
    np.testing.assert_allclose(c.numpy(), expected)

  def test_matrix_multiply(self):
    # Create matrices
    a = Tensor([[1, 2], [3, 4]], device="I915")
    b = Tensor([[5, 6], [7, 8]], device="I915")
    
    # Perform matrix multiplication
    c = a @ b
    
    # Check the result
    expected = np.array([[19, 22], [43, 50]])
    np.testing.assert_allclose(c.numpy(), expected)

if __name__ == "__main__":
  unittest.main()