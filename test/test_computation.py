import unittest
import torch
import jax, jax.numpy as jnp
from memory_efficient_attention import efficient_dot_product_attention_pt, efficient_dot_product_attention_jax
from flax.linen.attention import dot_product_attention
from memory_efficient_attention.utils import dynamic_slice

efficient_dot_product_attention_jax = jax.jit(efficient_dot_product_attention_jax, static_argnames=('mask_calc_fn', 'bias_calc_fn', 'weights_calc_fn'))

class ComputationTest(unittest.TestCase):
    @staticmethod
    def data():
        b = 8
        key = jax.random.PRNGKey(0)
        Qb = jax.random.uniform(key, (1, b, 128, 16, 8), dtype=jnp.float32)
        Kb = jax.random.uniform(key, (1, b, 128, 16, 8), dtype=jnp.float32)
        Vb = jax.random.uniform(key, (1, b, 128, 16, 8), dtype=jnp.float32)
        Mb = jax.random.uniform(key, (1, b, 16, 128, 128)) > 0.5
        Bb = jax.random.uniform(key, (1, b, 16, 128, 128), dtype=jnp.float32) / 100

        # calc_fn bias & mask
        def mask_calc_fn_jax(query_offset, key_offset, mask_chunk, attn_weights, MbBb):
            if MbBb is not None:
                Mb, Bb = MbBb
                return jax.lax.dynamic_slice(Mb, tuple([0] * (Mb.ndim - 2)) + (query_offset, key_offset),
                    slice_sizes=tuple(Mb.shape[:-2]) + (attn_weights.shape[-3], attn_weights.shape[-1]))
            else:
                return mask_chunk
        def bias_calc_fn_jax(query_offset, key_offset, bias_chunk, attn_weights, MbBb):
            if MbBb is not None:
                Mb, Bb = MbBb
                return jax.lax.dynamic_slice(Bb, tuple([0] * (Bb.ndim - 2)) + (query_offset, key_offset),
                    slice_sizes=tuple(Bb.shape[:-2]) + (attn_weights.shape[-3], attn_weights.shape[-1]))
            else:
                return bias_chunk
        def weights_calc_fn_jax(query_offset, key_offset, attn_weights, MbBb):
            if MbBb is not None:
                bias = bias_calc_fn_jax(query_offset, key_offset, None, attn_weights, MbBb)
                bias = jnp.einsum('...hqk->...qhk', bias)
                attn_weights = attn_weights + bias

                mask = mask_calc_fn_jax(query_offset, key_offset, None, attn_weights, MbBb)
                big_neg = jnp.finfo(attn_weights.dtype).min
                mask = jnp.einsum('...hqk->...qhk', mask)
                attn_weights = jnp.where(mask, attn_weights, big_neg)

            return attn_weights
        def mask_calc_fn_pt(query_offset, key_offset, mask_chunk, attn_weights, MbBb):
            if MbBb is not None:
                Mb, Bb = MbBb
                return dynamic_slice(torch.tensor(Mb.to_py()), tuple([0] * (Mb.ndim - 2)) + (query_offset, key_offset),
                    tuple(Mb.shape[:-2]) + (attn_weights.shape[-3], attn_weights.shape[-1]))
            else:
                return mask_chunk
        def bias_calc_fn_pt(query_offset, key_offset, bias_chunk, attn_weights, MbBb):
            if MbBb is not None:
                Mb, Bb = MbBb
                return dynamic_slice(torch.tensor(Bb.to_py()), tuple([0] * (Bb.ndim - 2)) + (query_offset, key_offset),
                    tuple(Bb.shape[:-2]) + (attn_weights.shape[-3], attn_weights.shape[-1]))
            else:
                return bias_chunk
        def weights_calc_fn_pt(query_offset, key_offset, attn_weights, MbBb):
            if MbBb is not None:
                bias = bias_calc_fn_pt(query_offset, key_offset, None, attn_weights, MbBb)
                bias = torch.einsum('...hqk->...qhk', bias)
                attn_weights = attn_weights + bias

                mask = mask_calc_fn_pt(query_offset, key_offset, None, attn_weights, MbBb)
                big_neg = torch.finfo(attn_weights.dtype).min
                big_neg = torch.tensor(big_neg, dtype=torch.float32)
                mask = torch.einsum('...hqk->...qhk', mask)
                attn_weights = torch.where(mask, attn_weights, big_neg)
            return attn_weights

        return [
            ('broadcasting simple mask and bias',
             (Qb, Kb, Vb, Mb[:,:,:1,:1,:1], Bb[:,:,:1,:1,:1], dict(), None)),
            ('full mask and bias, negates memory savings',
             (Qb, Kb, Vb, Mb, Bb, dict(), None)),
            ('broadcasting mask and bias passed through callbacks',
             (Qb, Kb, Vb, Mb[:,:,:1,:1,:1], Bb[:,:,:1,:1,:1], dict(
                pt_mask=mask_calc_fn_pt,
                pt_bias=bias_calc_fn_pt,
                pt_weights=weights_calc_fn_pt,
                jax_mask=mask_calc_fn_jax,
                jax_bias=bias_calc_fn_jax,
                jax_weights=weights_calc_fn_jax,
             ), None)),
            ('mask and bias generated per-chunk by callbacks',
             (Qb, Kb, Vb, Mb, Bb, dict(
                pt_mask=mask_calc_fn_pt,
                pt_bias=bias_calc_fn_pt,
                jax_mask=mask_calc_fn_jax,
                jax_bias=bias_calc_fn_jax,
              ), (Mb, Bb))),
            ('mask and bias manually applied by custom callback',
             (Qb, Kb, Vb, Mb, Bb, dict(
                pt_weights=weights_calc_fn_pt,
                jax_weights=weights_calc_fn_jax,
             ), (Mb, Bb))),
            ('no mask nor bias',
             (Qb, Kb, Vb, None, None, dict(), None)),
        ]

    @staticmethod
    def calc_pt(data):
        Qb, Kb, Vb, Mb, Bb, Cf, Cb = data
        Qbt = torch.tensor(Qb.to_py(), requires_grad=True)
        Kbt = torch.tensor(Kb.to_py(), requires_grad=True)
        Vbt = torch.tensor(Vb.to_py(), requires_grad=True)
        if Mb is not None and Cb is None:
            Mbt = torch.tensor(Mb.to_py(), requires_grad=False)
        else:
            Mbt = None
        if Bb is not None and Cb is None:
            Bbt = torch.tensor(Bb.to_py(), requires_grad=False)
        else:
            Bbt = None
        Mf, Bf, Wf = Cf.get('pt_mask'), Cf.get('pt_bias'), Cf.get('pt_weights')
        return efficient_dot_product_attention_pt(Qbt, Kbt, Vbt, Mbt, Bbt,
                                                  mask_calc_fn=Mf, bias_calc_fn=Bf,
                                                  weights_calc_fn=Wf, calc_fn_data=Cb).detach().numpy()

    @staticmethod
    def calc_jax(data):
        Qb, Kb, Vb, Mb, Bb, Cf, Cb = data
        Mf, Bf, Wf = Cf.get('jax_mask'), Cf.get('jax_bias'), Cf.get('jax_weights')
        if Cb is not None:
            Mb, Bb = None, None
        return jnp.asarray(efficient_dot_product_attention_jax(Qb, Kb, Vb, Mb, Bb,
                                                               mask_calc_fn=Mf, bias_calc_fn=Bf,
                                                               weights_calc_fn=Wf, calc_fn_data=Cb))

    @staticmethod
    def calc_flax(data):
        Qb, Kb, Vb, Mb, Bb, Cf, Pd = data
        return jnp.asarray(dot_product_attention(Qb, Kb, Vb, Bb, Mb))

    def test_pt(self):
        for msg, data in ComputationTest.data():
            with self.subTest(msg=msg):
                res_pt = ComputationTest.calc_pt(data)
                res_flax = ComputationTest.calc_flax(data)
                self.assertTrue(jnp.allclose(res_pt, res_flax))

    def test_jax(self):
        for msg, data in ComputationTest.data():
            with self.subTest(msg=msg):
                res_jax = ComputationTest.calc_jax(data)
                res_flax = ComputationTest.calc_flax(data)
                self.assertTrue(jnp.allclose(res_jax, res_flax))

    def test_jax_and_pt(self):
        for msg, data in ComputationTest.data():
            with self.subTest(msg=msg):
                res_pt = ComputationTest.calc_pt(data)
                res_jax = ComputationTest.calc_jax(data)
                res_flax = ComputationTest.calc_flax(data)
                self.assertTrue(jnp.allclose(res_pt, res_jax))
                self.assertTrue(jnp.allclose(res_pt, res_flax))
                self.assertTrue(jnp.allclose(res_jax, res_flax))


if __name__ == '__main__':
    unittest.main()
