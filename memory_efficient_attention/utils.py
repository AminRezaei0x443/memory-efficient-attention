import torch
import numpy as np


def dynamic_slice(x, starts, sizes):
    # start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - size_indices[i])
    starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))]
    indices = [slice(start, start + size) for start, size in zip(starts, sizes)]
    return x[indices]


def map_pt(f, xs):
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    carry, y = f(carry, xs[0])
    if type(y) is tuple:
        ys = tuple(([y_] for y_ in y))
        for x in xs[1:]:
            carry, y = f(carry, x)
            for ys_, y_ in zip(ys, y):
                ys_.append(y_)
        return carry, tuple((torch.stack(ys_) for ys_ in ys))
    else:
        ys = [y]
        for x in xs[1:]:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, torch.stack(ys)
