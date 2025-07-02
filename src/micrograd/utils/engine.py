## STARTING OF Pytorch-replice(confuses github copilot) ENGINE
import math
import numpy as np

from typing import Any

class Value:
    pass

class Value:
    def __init__(self, data: Any, _children: tuple[Value] = (), _op: str = "", label:str = "", grad:float = 0.0):
        if isinstance(data, Value):
            raise TypeError("Value cannot be initialized with another Value instance")
        self.data = data
        self._children = _children
        self._op = _op
        self.label = label
        self.grad = grad
        self._backward = None

    def __repr__(self)->str:
        return f"Value(data={self.data :0.04f}, label={self.label})"#, _children={self._children}, _op='{self._op}')"
    
    def __add__(self, other:Value)->Value:
        if not isinstance(other, Value):
            other = Value(other)
        def _backward(parent_grad:float = 1.0):
                other.grad += (1.0* parent_grad)
                self.grad += (1.0 * parent_grad)

        out = Value(self.data + other.data, _children=(self, other), _op='+')
        out._backward = _backward
        return out
    
    def __sub__(self, other:Value)->Value:
        if not isinstance(other, Value):
            other = Value(other)
        def _backward(parent_grad:float = 1.0):
            self.grad += (1.0 * parent_grad)
            other.grad -= (1.0 * parent_grad)

        out = Value(self.data - other.data, _children=(self, other), _op='-')
        out._backward = _backward
        return out
    
    def __mul__(self, other:Any)->Value:
        if not isinstance(other, Value):
            other = Value(other)
        def _backward(parent_grad:float = 1.0):
            self.grad += (other.data*parent_grad)
            other.grad += (self.data*parent_grad)

        out = Value(self.data * other.data, _children=(self, other), _op='*')
        out._backward = _backward
        return out
    
    def __pow__(self, other:Any)->Value:
        if not isinstance(other, Value):
            other = Value(other)
        def _backward(parent_grad:float = 1.0):
            self.grad = (other.data* self.data**(other.data -1.0)) * parent_grad

        out = Value(self.data ** other.data, _children=(self, other), _op='**')
        out._backward = _backward
        return out
    
    def __rpow__(self, other:Any)->Value:
        return Value(other**self.data, _children=(Value(other), self), _op='**')

    def __radd__(self, other:Any)->Value:
        return self + other
    
    def __rsub__(self, other:Any)->Value:
        return self-other
    
    def __rmul__(self, other:Any)->Value:
        return self*other
        
    def __true_div__(self, other:Any)->Value:
        return self*other**-1

    def __rtruediv__(self, other:Any)->Value:
        return other*(self**-1)
    
    def sin(self)->Value:
        def _backward(parent_grad:float = 1.0):
            self.grad = np.cos(self.data)*parent_grad
        out = Value(np.sin(self.data), _children = (self,), _op = "sin")
        out._backward = _backward
        return out
    
    def cos(self)->Value:
        def _backward(parent_grad:float = 1.0):
            self.grad = -1.0*np.sin(self.data)*parent_grad
        out = Value(np.cos(self.data), _children = (self,), _op = "cos")
        out._backward = _backward
        return out
    
    def tanh(self)->Value:
        tanh = (1 - math.exp(-2*self.data))/(1+math.exp(-2*self.data))
        def _backward(parent_grad:float = 1.0):
            self.grad += (1-tanh**2)*parent_grad

        out = Value(tanh, _children = (self,), _op = "tanh")
        out._backward=_backward
        return out
    
    def backward(self)->None:
        def _backward(node:Value):
            if not node._children:
                return None
            node._backward(parent_grad=node.grad)
            for child in node._children:
                _backward(child)
        self.grad = 1.0
        _backward(self)
                
