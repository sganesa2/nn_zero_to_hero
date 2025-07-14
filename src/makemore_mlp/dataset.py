#Create dataset
import torch
import random

from dataclasses import dataclass, field
from pathlib import Path

def stoi()->dict:
    start_id = 97
    stoi_dict = {chr(i):i-start_id+1 for i in range(start_id,start_id+26)}
    stoi_dict["."] = 0
    return stoi_dict

def itos()->dict:
    stoi_dict =stoi()
    return {v:k for k,v in stoi_dict.items()}

@dataclass
class MLPDataset:
    file_name:str
    train_size:int
    test_size:int
    dev_size:int

    context_size:int

    xs: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    ys: torch.Tensor = field(default_factory=lambda: torch.tensor(0))

    @property
    def trainset(self)->tuple[torch.Tensor, torch.Tensor]:
        xs, ys = self.xs[:self.train_size], self.ys[:self.train_size]
        return xs,ys

    @property
    def testset(self)->tuple[torch.Tensor, torch.Tensor]:
        i = self.train_size
        xs, ys = self.xs[i: self.test_size], self.ys[i: self.test_size]
        return xs,ys

    @property
    def devset(self)->tuple[torch.Tensor, torch.Tensor]:
        i = self.train_size + self.test_size
        xs, ys = self.xs[i: self.dev_size], self.ys[i: self.dev_size]
        return xs,ys

    def create_complete_dataset(self)->tuple[torch.Tensor, torch.Tensor]:
        with open(Path(f"./{self.file_name}"), 'r') as f:
            names = f.read().splitlines()
            random.shuffle(names)
        stoi_dict = stoi()

        X,Y = [],[]
        for name in names:
            chrs = list(name) + ["."]
            context = [0]*self.context_size
            for i in range(len(name)):
                X.append(context)
                Y.append(stoi_dict[chrs[i]])
                context = context[1:] + [stoi_dict[chrs[i]]]

        self.xs, self.ys = torch.tensor(X), torch.tensor(Y)
        return self.xs, self.ys
