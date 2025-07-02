from copy import deepcopy

from utils.mlp import MLP

def gradient_descent(mlp:MLP, dataset:dict, max_iters:int = 1000, h:float = 0.01)->MLP:
    loss = sum((mlp(inp)-pred)**2 for inp, pred in zip(*dataset.values())); loss.label = 'loss'
    best_loss = (loss,deepcopy(mlp.parameters))
    for _ in range(max_iters):
        for param in mlp.parameters:
            param.data-= h*param.grad
            param.grad=0.0
        loss.backward()
        loss = sum((mlp(inp)-pred)**2 for inp, pred in zip(*dataset.values())); loss.label = 'loss'
        if loss.data>best_loss[0].data:
            return
        best_loss = (loss,deepcopy(mlp.parameters))
    return mlp