# import matplotlib
from matplotlib import pyplot as pp
import torch as pt


def example(nb=1, ni=11, nx=13, ny=17):    # prime numbers help to understand final tensor shape, see: prime-factorization
    v = pt.empty(nb, ni, nx, ny)

    # example_000()
    pp.figure(1).clf()

    pp.subplot(2, 2, 1)
    # pp.figure(3).clf()
    v = pt.zeros_like(v)
    v[:] = v + pt.arange(ni)[None, :, None, None]  # color by input channel
    i = im2col(v, kernel_size=5)
    pp.imshow(i[0, :, :])
    pp.title('A) Channel Slice')
    pp.xticks([])
    pp.yticks([])

    pp.subplot(2, 2, 2)
    # pp.figure(4).clf()
    v = pt.zeros_like(v)
    v[:] = v + pt.arange(nx)[None, None, :, None]  # color by X location
    i = im2col(v, kernel_size=5)
    pp.imshow(i[0, :, :])
    pp.title('B) X Slice')
    pp.xticks([])
    pp.yticks([])

    pp.subplot(2, 2, 3)
    # pp.figure(5).clf()
    v = pt.zeros_like(v)
    v[:] = v + pt.arange(ny)[None, None, None, :]  # color by Y Location
    i = im2col(v, kernel_size=5)
    pp.imshow(i[0, :, :])
    pp.title('C) Y Slice')
    pp.xticks([])
    pp.yticks([])

    pp.subplot(2, 2, 4)
    # pp.figure(6).clf()
    v = pt.zeros_like(v)
    v[0, 6, 8, 9] = 1.0  # color a single pixel on input
    i = im2col(v, kernel_size=5)  # notice 25x duplication
    pp.imshow(i[0, :, :], interpolation='none', aspect='equal')
    pp.title('D) Single Pixel from Input')
    pp.xticks([])
    pp.yticks([])

    pp.tight_layout()


def im2col(input, kernel_size, dilation=1, padding=None, stride=1):
    assert input.ndim == 4

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    padding = [i // 2 for i in kernel_size] if padding is None else padding
    bs, ch, *sz_in = input.shape

    # calculate output size, info: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # output_size = [(i + 2 * p - d * (k - 1) - 1) // s + 1 for i, p, d, k, s in zip(sz_in, [padding]*2, [dilation]*2, [kernel_size]*2, [stride]*2)]
    unfold = pt.nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    # the IM2COL representation of the 2d-spatial input
    b = unfold(input).transpose(-1, -2)

    return b


if __name__ == '__main__':
    example()
