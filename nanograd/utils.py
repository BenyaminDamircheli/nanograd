def broadcasted_axis(left, right):
    """
    mlx uses broadcasting before performing array ops
    this function determines which axes on either arrays will be broadcasted
    in order to calculate gradients along those axes.

    broadcasting implicitly expands the arrays to match eachother so that
    array ops can be performed element-wise. The elements are not necessarily copied,
    which makes this operation efficient.

    example:
    >>> left.shape = (3, 1)
    >>> right.shape = (1, 4)
    >>> broadcasted_axis(left, right)     # ((1, ), (0, ))

    here the second axis for left, and first axis for right will be broadcasted
    """
    
    leftdim = len(left)
    rightdim = len(right)
    maxdim = max(leftdim, rightdim)

    leftshape_new = (1, ) * (maxdim - leftdim) + left
    rightshape_new = (1, ) * (maxdim - rightdim) + right

    assert len(leftshape_new) == len(rightshape_new), "left and right must have the same number of dimensions"

    left_axes, right_axes = [], []

    for i in range(len(leftshape_new)):
        if leftshape_new[i] > rightshape_new[i]:
            right_axes.append(i)
        elif rightshape_new[i] > leftshape_new[i]:
            left_axes.append(i)

    return tuple(left_axes), tuple(right_axes)