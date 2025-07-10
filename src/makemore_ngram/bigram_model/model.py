#Create single layer neural net with 27 neurons using torch
import torch
import torch.nn.functional as F
from dataset import Dataset, itos

class BigramModel:
    def __init__(self, nin:int, nout:int)->None:
        self.generator = torch.Generator().manual_seed(6385189022)
        self.W = torch.randn((nin,nout), requires_grad= True, generator=self.generator) #27 neurons in 1st hidden layers; takes in 27 inputs
        self.a_index = 97
        self.nll_loss = 0.0
    
    def forward(self, inp_vectors:torch.Tensor)->torch.Tensor:
        logits = inp_vectors@self.W
        dim = 1 if len(logits.shape)>1 else 0
        p = logits.softmax(dim=dim, dtype=torch.float32)
        return p
    
    def gradient_descent(self, inp_vectors:torch.Tensor, outputs:torch.Tensor, reg_factor:float, iters:int, h:float = 10.0)->None:
        for _ in range(iters):
            #forward pass
            p = self.forward(inp_vectors=inp_vectors)

            #compute loss
            self.nll_loss = -p[torch.arange(outputs.nelement()), outputs].log().mean() + reg_factor*(self.W**2).mean()
            
            #weight update
            self.nll_loss.backward()
            self.W.data -= h*self.W.grad
            self.W.grad = None

if __name__=="__main__":
    dataset_obj = Dataset("dataset.txt", 156890, 19611,19611)
    dataset = dataset_obj.get_complete_dataset()
    xenc,labels = dataset_obj.enc_train_set

    bigram_model = BigramModel(27,27)
    bigram_model.gradient_descent(xenc, labels, 0.001, 1000)

    for _ in range(5):
        c = ""
        word = ""
        index = 0
        while c!=".":
            word +=c
            i_vec = F.one_hot(torch.tensor(index), num_classes=27).float()
            p = bigram_model.forward(inp_vectors=i_vec)
            index = torch.multinomial(p, num_samples=1).item()
            c = itos()[index]
        print(word)