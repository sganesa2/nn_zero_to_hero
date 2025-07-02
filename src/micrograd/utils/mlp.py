#Replicating pytorch's nn.Module
import random

from utils.engine import Value

class Neuron:
    def __init__(self, nin:int, neuron_no:int, layer_no:int)->None:
        self.w = [Value(random.uniform(0,1), label = f"w{i}{neuron_no}{layer_no}") for i in range(nin)]
        self.b = Value(random.uniform(0,1), label = f"b{neuron_no}{layer_no}")

    def __repr__(self)->str:
        return f"Neuron(w= {self.w}, b= {self.b})"
    
    def __call__(self, inp:list[Value],neuron_no:int, layer_no:int)->Value:
        res = sum([w*x for w,x in zip(self.w, inp)], self.b)
        res.label = f"dot_prod_{neuron_no}{layer_no}"
        out = res.tanh()
        out.label = f"n{neuron_no}{layer_no}"
        return out
    
    @property
    def parameters(self)->list[Value]:
        return self.w + [self.b]

class MLP:
    def __init__(self, n_layers:list[int]):
        self.n_layers = n_layers
        self.hidden_layers = []
        self.out = []

    def __repr__(self)->str:
        return f"MLP(hidden_layers= {self.hidden_layers}, out= {self.out})"
    
    def create_hidden_layer(self, nin:int, layer_no:int, neuron_count:int)->None:
        self.hidden_layers.append([Neuron(nin, j, layer_no) for j in range(neuron_count)])


    def create_all_hidden_layers(self, inp:list[float])->None:
        inp = [Value(x, label = f"x{i}") for i,x in enumerate(inp)]
        self.n_layers = [len(inp),*self.n_layers]
        for i, neuron_count in enumerate(self.n_layers[1:]):
            self.create_hidden_layer(self.n_layers[i],i, neuron_count)
    
    def __call__(self, inp:list[Value])->Value|list[Value]:
        if not self.hidden_layers:
            self.create_all_hidden_layers(inp)
        input_vec = inp
        for i, hl in enumerate(self.hidden_layers):
            input_vec = [neuron(input_vec, j, i) for j, neuron in enumerate(hl)]
            
        self.out = input_vec[0] if len(input_vec)==1 else input_vec
        return self.out
    
    @property
    def parameters(self)->list[Value]:
        return [p for hl in self.hidden_layers for n in hl for p in n.parameters]