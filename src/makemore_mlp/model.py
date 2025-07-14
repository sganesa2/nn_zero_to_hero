#Create the MLP(variant version in Bengio et al 2003 paper)
import torch
import torch.nn.functional as F
from typing import Optional

class MLP:
    def __init__(self, context_size:int, feature_dims:int, h:int, total_chrs:int = 27)->None:
        self.n = context_size
        self.m = feature_dims
        self.h = h

        self.generator = torch.Generator().manual_seed(6385189022)
        self.C = torch.randn((total_chrs, feature_dims), generator=self.generator, requires_grad=True)

        self.H = torch.randn((self.n*self.m,self.h), generator=self.generator, requires_grad=True)
        self.b1 = torch.randn(self.h, generator= self.generator) #Hidden layer neurons biases for input sequence of feature vectors

        self.W1 = torch.randn((self.h, total_chrs), generator= self.generator, requires_grad=True) #Output layer weights for hidden layer neurons
        self.W2 = torch.randn((self.n*self.m, total_chrs), generator= self.generator, requires_grad=True) #Output layer weights for INPUT VECTOR
        self.b2 = torch.randn(total_chrs, generator= self.generator) #Output layer neurons biases for input sequence fo feature vectors

        self.cross_entropy_loss:float = 0.0
        
    def lookup_embedding(self, xs:torch.Tensor)->torch.Tensor:
        embed = self.C[xs]
        return embed.view(-1,self.n*self.m)
    
    def forward(self, xs:torch.Tensor)->torch.Tensor:
        inp_vector = self.lookup_embedding(xs)
        hidden_op = torch.tanh(inp_vector@self.H + self.b1)
        hidden_to_out_op = hidden_op@self.W1
        inp_and_out_op = inp_vector@self.W2 + self.b2
        logits = hidden_to_out_op + inp_and_out_op
        return logits
    
    def gradient_descent(self, xs:torch.Tensor, ys:torch.Tensor, no_of_iters:int, h:float, reg_factor:float)->None:
        for _ in range(no_of_iters):
            #forward pass
            logits = self.forward(xs)

            #compute loss
            loss = F.cross_entropy(logits, ys, label_smoothing=reg_factor)

            self.C.grad = None; self.H.grad = None; self.W1.grad = None; self.W2.grad = None

            loss.backward()
            self.cross_entropy_loss = loss

            #update weight matrices and embedding lookup table
            self.C.data -= h*self.C.grad
            self.H.data -= h*self.H.grad
            self.W1.data -= h*self.W1.grad
            self.W2.data -= h*self.W2.grad

