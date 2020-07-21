import unittest
import cupy as cp


from pool_layer import PoolLayer


class TestPoolLayer(unittest.TestCase):

    def test_forward_propagation(self):
        X1 = cp.random.rand(10, 4, 4, 3)
        L = PoolLayer(X1.shape, f=2, stride=2, mode='max')
        Z1 = L.forward_naive(X1)
        Z2 = L.forward(X1)
        self.assertAlmostEqual(float(Z1[1, 0, 0, 0]), float(Z2[1, 0, 0, 0]))
        self.assertTrue((Z1 == Z2).all())
        self.assertEqual(Z1.shape, Z2.shape)

    def test_backward_propagation(self):
        X1 = cp.random.rand(10, 4, 4, 3)
        L = PoolLayer(X1.shape, f=2, stride=2, mode='max')
        dA = L.forward(X1)
        dX1 = L.backward_naive(dA)
        dX2 = L.backward(dA)
        self.assertEqual(dX1.shape, dX2.shape)
        self.assertTrue((dX1 == dX2).all())
