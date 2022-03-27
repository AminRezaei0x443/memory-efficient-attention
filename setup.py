from setuptools import setup, find_packages

setup(
    name='memory-efficient-attention',
    version='0.1.3',
    description='Memory Efficient Attention (O(sqrt(n)) for Jax and PyTorch',
    license='MIT',
    packages=find_packages(),
    author='Amin Rezaei',
    author_email='AminRezaei0x443@gmail.com',
    keywords=['attention', 'pytorch', 'jax'],
    url='https://github.com/AminRezaei0x443/memory-efficient-attention',
    install_requires=['numpy'],
    extras_require={
        'jax': ['jax'],
        'torch': ['torch'],
        'testing': ['jax', 'torch', 'flax']
    }
)
