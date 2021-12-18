import unittest
import torch
import numpy as np
from memory_efficient_attention import efficient_dot_product_attention_pt, efficient_dot_product_attention_jax
from flax.linen.attention import dot_product_attention


class ComputationTest(unittest.TestCase):
    @staticmethod
    def data():
        b = 8
        Qb = np.random.rand(1, b, 128, 16, 8).astype("float32")
        Kb = np.random.rand(1, b, 128, 16, 8).astype("float32")
        Vb = np.random.rand(1, b, 128, 16, 8).astype("float32")
        Mb = np.random.rand(1, b, 16, 128, 128) > 0.5
        Bb = np.random.rand(1, b, 16, 128, 128).astype("float32") / 100
        return Qb, Kb, Vb, Mb, Bb

    @staticmethod
    def calc_pt(data):
        Qb, Kb, Vb, Mb, Bb = data
        Qbt = torch.tensor(Qb, requires_grad=True)
        Kbt = torch.tensor(Kb, requires_grad=True)
        Vbt = torch.tensor(Vb, requires_grad=True)
        Bbt = torch.tensor(Bb, requires_grad=True)
        Mbt = torch.tensor(Mb)
        return efficient_dot_product_attention_pt(Qbt, Kbt, Vbt, Mbt, Bbt).detach().numpy()

    @staticmethod
    def calc_jax(data):
        Qb, Kb, Vb, Mb, Bb = data
        return np.asarray(efficient_dot_product_attention_jax(Qb, Kb, Vb, Mb, Bb))

    @staticmethod
    def calc_flax(data):
        Qb, Kb, Vb, Mb, Bb = data
        return np.asarray(dot_product_attention(Qb, Kb, Vb, Bb, Mb))

    def test_pt(self):
        data = ComputationTest.data()
        res_pt = ComputationTest.calc_pt(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_pt, res_flax))

    def test_jax(self):
        data = ComputationTest.data()
        res_jax = ComputationTest.calc_jax(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_jax, res_flax))

    def test_jax_and_pt(self):
        data = ComputationTest.data()
        res_pt = ComputationTest.calc_pt(data)
        res_jax = ComputationTest.calc_jax(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_pt, res_jax))
        self.assertTrue(np.allclose(res_pt, res_flax))
        self.assertTrue(np.allclose(res_jax, res_flax))


if __name__ == '__main__':
    unittest.main()
