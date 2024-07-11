# This is my implementation of Andrej Karpathy's micrograd.
# This file uses numpy / MLX, which is a machine learning library
# made to take advantage of the apple silicon neural engine.

import numpy as np
import mlx.core as mx

from typing import *

# for mlx broadcasting
from ..utils import broadcasted_axis

# Array type for signatures
Array = Union[np.ndarray, mx.array]

class Tensor:
    """
    Tensor implementation. Holds elements of the same dimension type.
    """
    def __init__(
        self,
        data: Union[Array, Any]
        dtype: none,
        op = none,
        children: tuple = (),
        require_grad: bool = True,
        use_np: bool = False,
    ) -> None:

    # if you have mlx installed it will use the apple nueral engine.
    self._d = get_device(device = "gpu" if not use_np else "cpu")
    self.dtype = dtype or self._d.float32

    # data in tensor
    self.data = (
        self._d.array(data, dtype=self.dtype) # create new array on the device if data is an Array type
        if isinstance(data, Array)
        else data.astype(dtype=self.dtype) # convert to dtype
    )

    # Set of previous tensors that require gradient calc. 
    # Also the operation that created this current tensor.
    self.prev = set([c for c in children if c.require_grad])
    self.op = op

    # gradient of this tensor
    self.requires_grad = require_grad
    self.grad = self._d.zeros_like(self.data) if require_grad else None
    self.grad_function = None

    # shape and number of dimensions of this tensor
    self.shape = self.data.shape
    self.ndim = len(self.shape)

    def get_device(self, device: str = "gpu"):
        if device == "gpu" return mx;
        if device == "cpu" return np;
        raise ValueError(f"Unknown device: {device}")
    
    def set_requires_grad(self, requires_grad:bool):
        if not isinstance(requires_grad, bool):
            raise ValueError(f"requires_grad must be a boolean, got {type(requires_grad)}")
        
        if self.requires_grad is None and requires_grad:
            self.grad = mx.zeros_like(self.data)
        
        self.requires_grad = requires_grad
    
    def backward(self):
        """
        sorts the computational graph topologically. Ensures that when we are calculating
        a gradient, all other necessary gradients have been calculated.
        
        runs the grad function from last node back to the first.

        ** BACKPROPAGATION **
        """
        order = []

        visited = set()
        recursion_stack = set()

        def topo_sort(curr: "Tensor"):
            if curr in rectursion_stack:
                raise Error("Graph contains cycle")
            
            if curr not in visited:
                visited.add(curr)
                recursion_stack.add(curr)

                for child in curr.prev:
                    topo_sort(child)
                
                recursion_stack.remove(curr)
                order.append(curr)
            
        topo_sort(self)

        # the grad with respect to itself is 1.
        self.grad = self._d._ones_like (self.data)

        # gradient on each prev node. Backpropagation!
        for node in reversed(order):
            if node.grad_function is not None:
                node.grad_function()

    
    # -------------------- Unary Operations ------------------
    # One variable stuff. Not two or more things being added/subtracted/multiplied/divided.

    def __neg__(self):
        return self * -1
    
    def half(self):
        """
        Converts grads to half the precision. float32  -> float16.
        """
        if self.dtype = float32:
            out = Tensor(
                self.dtype = self._d.float16
                children = (self, )
                op = "half"
            )
        
            if self.requires_grad:
                # gradients just flow from out (next node in comp graph) to self.
                def half_backward():
                    self.grad += out.grad
                
                out.grad_function = half_backward
                out.set_requires_grad(True)
            
            return out
        
        else:
            raise ValueError(f"Cannot half a tensor of dtype {self.dtype}")

    def T(self, axes: Iterable = None):
        """
        Transposes the tensor along a given axis.
        """

        out = Tensor(
            data = self._d.transpose(self.data, axes = axes),
            children = (self, ),
            op = "T"
        )

        if self.requires_grad:
            # gradients just flow from out (next node in comp graph) to self.
            def T_backward():
                self.grad += self._d.transpose(out.grad, axes = axes)
            
            out.grad_function = T_backward
            out.set_requires_grad(True)
        
        return out

    
    def exp(self):
        """
        Exponentiates the tensor. Power of e.
        """

        out.Tensor(
            data = self._d.exp(self.data),
            children = (self, )
            op = "exp"
        )

        if self.requires_grad:
            # Here gradient is just the original data times the gradient of the next node.
            # because d[e^x]/dx = e^x
            def exp_backward():
                self.grad += self.data * out.grad
            
            out.grad_function = exp_backward
            out.set_requires_grad(True)
        
        return out

    # -------------------- Binary Operations ------------------

    def __add__(self, other: "Tensor"):
        """
        elementwise addition (takes broadcasting into account)
        """

        # for when trying to add scalars
        if isinstance(other, (int, float)):
            out = Tensor(
                data = self.data + other,
                children = (self, ) #just self because scalar is not a tensor
                op = "add"
            )
            if self.requires_grad:
                def add_backward_scalar():
                    self.grad += out.grad
            
                out.grad_function = add_backward
                out.set_requires_grad(True)

        return out

        else:
            other = other if isinstance(other, Tensor) else Tensor(other)
            out = Tensor(
                data = self.data + other.data, 
                children = (self, other),
                op = "add"
            )

            if self.requires_grad == False and other.requires_grad == False:
                return out
            
            if self.shape == other.shape:
               
                def add_backward_same():
                    if self.requires_grad:
                        self.grad += out.grad
                    
                    if other.requires_grad:
                        other.grad += out.grad
                
                out.grad_function = add_backward_same
            
            else:
                # since diff shapes, broadcast occurs
                # gradient is summed up across all of the broadcast axes
                # since the out Tensor is result of broadcasting and addition
                # in essence, broadcasted axes are copied and added, so gradients from 
                # all the copies should be added

                laxis, raxis = broadcasted_axis(self.data.shape, other.data.shape)

                def add_backward_diff():
                    if self.requires_grad:
                        self.grad += self._d.reshape(
                            mx.sum(out.grad, axis = laxis), self.shape
                        )
                    
                    if other.requires_grad:
                        other.grad += other._d.reshape(
                            mx.sum(out.grad, axis = raxis), other.shape
                        )

                out.grad_function = add_backward_diff
            
            out.set_requires_grad(True)
            return out


    # Double check this later
    def __mul__(self, other: "Tensor"):
        """
        elementwise multiplication (takes broadcasting into account)
        """               
        if isinstance(other, (int, float)):
            out = Tensor(
                data = self.data * other,
                children = (self,),
                op = "mul"
            )
        
        if self.requires_grad == False or other.requires_grad == False:
            return out
        
        if self.shape == other.shape:
            def mul_backward_same():
                if self.requires_grad:
                    self.grad += out.grad * 1
                if other.requires_grad:
                    other.grad += out.grad * 1
        
        else:
            
        
        







            














    

