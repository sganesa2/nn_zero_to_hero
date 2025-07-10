import torch
import torch.nn.functional as F

from dataset import Dataset, itos

class Trigram:
    def __init__(self, nin:int, neuron_count:int)->None:
        self.generator = torch.Generator().manual_seed(6385189022)
        self.W = torch.randn((nin, neuron_count), generator=self.generator, requires_grad=True)
        self.cross_entropy_loss:float = 0.0
    
    def forward(self, xenc:torch.Tensor)->torch.Tensor:
        logits = xenc@self.W
        return logits

    def gradient_descent(self, xenc:torch.Tensor, labels:torch.Tensor, iters:int, h:float = 5.0)->None:
        for _ in range(iters):
            #forward pass
            logits = self.forward(xenc)

            #compute loss
            self.cross_entropy_loss = F.cross_entropy(logits, torch.tensor(labels), label_smoothing=0.1)

            #update weights
            self.cross_entropy_loss.backward()
            self.W.data-=h*self.W.grad
            self.W.grad = None

if __name__=="__main__":
    dataset_obj = Dataset("dataset.txt", 156890, 19611,19611)
    dataset = dataset_obj.get_complete_dataset()
    xenc,labels = dataset_obj.enc_train_set

    trigram_model = Trigram(27*27,27)
    trigram_model.gradient_descent(xenc, labels, 1000)

    for _ in range(5):
        c = ""
        index = torch.multinomial(torch.randn(27), num_samples=1).item()
        word = itos()[index]
        while c!=".":
            word +=c
            i_vec = F.one_hot(torch.tensor(index), num_classes=27).float()
            #forward pass
            logits = trigram_model.forward(inp_vectors=i_vec)
            dim = 1 if len(logits.shape>1) else 0
            p = logits.softmax(dim)

            index = torch.multinomial(p, num_samples=1).item()
            c = itos()[index]
        print(word)