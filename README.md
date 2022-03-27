# Memory Efficient Attention
[![arXiv](https://img.shields.io/badge/arXiv-2112.05682v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2112.05682v2)
[![PyPI version](https://img.shields.io/badge/memory--efficient--attention-0.1.3-informational?style=flat-square&color=C51BA3)](https://pypi.org/project/memory-efficient-attention/)

This is **unofficial** implementation of [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682v2) for Jax and PyTorch.

Implementation is almost same as the one proposed in the paper, with additional **masking and adding bias compatibility**, **batch dimensions support** and **PyTorch implementation**. 
For computing attention, the proposed method requires only O(sqrt(n)) memory, and the provided functions can be used as a drop-in replacement for attention calculation.

**Important Note:** This implementation is a trade-off between memory requirements and runtime, so you should adjust `key_chunk_size` and `query_chunk_size` parameters to achieve the best configuration for your usecase. Here is a note from the paper's authors:

>While a constant chunk size for the queries and a chunk size of sqrt(n)
>for the keys and values is optimal for memory
>consumption, the runtime is also affected by the choice of chunk size
>in practice, which is heavily affected by the
>choice of hardware. Ultimately, we have to leave this trade-off to the
>programmer, and expose the chunk sizes as
>arguments query_chunk_size and key_chunk_size. In Figure 1 we provide default values for the chunk sizes that
lead to minimal runtime impact (on TPUv2), while still providing significant memory savings.


## Quick Start
1. Install the library

```bash
# for Jax
pip install memory-efficient-attention[jax]
# for PyTorch
pip install memory-efficient-attention[torch]
# for Running Tests
pip install memory-efficient-attention[testing]
```

2. Compute attention with the proper function

```python
import numpy as np
# for PyTorch
from memory_efficient_attention import efficient_dot_product_attention_pt
# or for Jax
from memory_efficient_attention import efficient_dot_product_attention_jax

# Random Data (batch dimensions are not necessary)
b = 8
query = np.random.rand(1, b, 128, 16, 8).astype("float32")
key = np.random.rand(1, b, 128, 16, 8).astype("float32")
value = np.random.rand(1, b, 128, 16, 8).astype("float32")
# optional, for casual tasks, ...
mask = np.random.rand(1, b, 16, 128, 128) > 0.5
bias = np.random.rand(1, b, 16, 128, 128).astype("float32") / 100

# Adjust chunk sizes        
efficient_dot_product_attention_jax(query, key, value, mask, bias, key_chunk_size=..., query_chunk_size=...)
```

## Citation
Please cite if this implementation helps your research. You can use the following BibTeX entry:

```bibtex
@misc{memory_efficient_attention,
	title = {Memory Efficient Attention},
	author = {Rezaei, Amin},
	howpublished = {\url{github.com/AminRezaei0x443/memory-efficient-attention}},
	year = {2021}
}
```
Also, for the paper:
```bibtex
@misc{rabe2021selfattention,
      title={Self-attention Does Not Need $O(n^2)$ Memory}, 
      author={Markus N. Rabe and Charles Staats},
      year={2021},
      eprint={2112.05682},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
