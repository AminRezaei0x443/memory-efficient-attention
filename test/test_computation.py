import unittest
import torch
import numpy as np
import jax, jax.numpy as jnp
from memory_efficient_attention import efficient_dot_product_attention_pt, efficient_dot_product_attention_jax
import math
from flax.linen.attention import dot_product_attention

def dot_product_attention_ex(query, key, value, bias, mask, return_attentions=False):
    if return_attentions is False:
        return dot_product_attention(query, key, value, bias, mask)
    attns = jnp.einsum('...qhd,...khd->...qhk', query / math.sqrt(key.shape[-1]), key, precision=jax.lax.Precision.HIGHEST)
    if bias is not None:
        bias = jnp.einsum('...hqk->...qhk', bias)
        attns += bias
    if mask is not None:
        big_neg = jnp.finfo(attns.dtype).min
        mask = jnp.einsum('...hqk->...qhk', mask)
        attns = jnp.where(mask, attns, big_neg)
    exp_attns = jax.nn.softmax(attns, axis=-1)
    if return_attentions:
        return jnp.einsum('...vhf,...qhv->...qhf', value, exp_attns), exp_attns
    else:
        return jnp.einsum('...vhf,...qhv->...qhf', value, exp_attns)

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
    def calc_pt(data, return_attentions=False):
        Qb, Kb, Vb, Mb, Bb = data
        Qbt = torch.tensor(Qb, requires_grad=True)
        Kbt = torch.tensor(Kb, requires_grad=True)
        Vbt = torch.tensor(Vb, requires_grad=True)
        Bbt = torch.tensor(Bb, requires_grad=True)
        Mbt = torch.tensor(Mb)
        out = efficient_dot_product_attention_pt(Qbt, Kbt, Vbt, Mbt, Bbt, return_attentions=return_attentions)
        if return_attentions:
            return out[0].detach().numpy(), out[1].detach().numpy()
        else:
            return out.detach().numpy()

    @staticmethod
    def calc_jax(data, return_attentions=False):
        Qb, Kb, Vb, Mb, Bb = data
        out = efficient_dot_product_attention_jax(Qb, Kb, Vb, Mb, Bb, return_attentions=return_attentions)
        if return_attentions:
            return np.asarray(out[0]), np.asarray(out[1])
        else:
            return np.asarray(out) 

    @staticmethod
    def calc_flax(data, return_attentions=False):
        Qb, Kb, Vb, Mb, Bb = data
        out = dot_product_attention_ex(Qb, Kb, Vb, Bb, Mb, return_attentions=return_attentions)
        if return_attentions:
            return np.asarray(out[0]), np.asarray(out[1])
        else:
            return np.asarray(out)

    def test_pt(self):
        data = ComputationTest.data()
        res_pt = ComputationTest.calc_pt(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_pt, res_flax))
        res_pt, attns_pt = ComputationTest.calc_pt(data, return_attentions=True)
        res_flax, attns_flax = ComputationTest.calc_flax(data, return_attentions=True)
        self.assertTrue(np.allclose(res_pt, res_flax))
        self.assertTrue(np.allclose(attns_pt, attns_flax))

    def test_jax(self):
        data = ComputationTest.data()
        res_jax = ComputationTest.calc_jax(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_jax, res_flax))
        res_jax, attns_jax = ComputationTest.calc_jax(data, return_attentions=True)
        res_flax, attns_flax = ComputationTest.calc_flax(data, return_attentions=True)
        self.assertTrue(np.allclose(res_jax, res_flax))
        self.assertTrue(np.allclose(attns_jax, attns_flax))

    def test_jax_and_pt(self):
        data = ComputationTest.data()
        res_pt = ComputationTest.calc_pt(data)
        res_jax = ComputationTest.calc_jax(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_pt, res_jax))
        self.assertTrue(np.allclose(res_pt, res_flax))
        self.assertTrue(np.allclose(res_jax, res_flax))
        res_pt, attns_pt = ComputationTest.calc_pt(data, return_attentions=True)
        res_jax, attns_jax = ComputationTest.calc_jax(data, return_attentions=True)
        res_flax, attns_flax = ComputationTest.calc_flax(data, return_attentions=True)
        self.assertTrue(np.allclose(res_pt, res_jax))
        self.assertTrue(np.allclose(res_pt, res_flax))
        self.assertTrue(np.allclose(res_jax, res_flax))
        self.assertTrue(np.allclose(attns_pt, attns_jax))
        self.assertTrue(np.allclose(attns_pt, attns_flax))
        self.assertTrue(np.allclose(attns_jax, attns_flax))

if __name__ == '__main__':
    unittest.main()
