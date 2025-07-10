from dataclasses import dataclass, field
from pathlib import Path
import random

import torch.nn.functional as F
import torch


def stoi()->dict:
    a_index = 97
    stoi = {chr(i):i-a_index+1 for i in range(a_index, a_index+26)}
    stoi["."] = 0
    return stoi

def itos()->dict:
    return {v:k for k,v in stoi().items()} 

@dataclass
class Dataset:
    file_name:str
    train_size:int
    test_size:int
    dev_size:int

    xpos: list[int] = field(default_factory=lambda:[])    
    xenc: torch.Tensor = field(default_factory=lambda:torch.tensor(0))
    ys: list[int] = field(default_factory=lambda:[])  
    labels:torch.Tensor = field(default_factory= lambda: torch.tensor(0))
    
    @property
    def enc_train_set(self)->tuple:
        xenc = self.xenc[:self.train_size]
        ys = self.labels[:self.train_size]

        return xenc,ys
    @property
    def enc_test_set(self)->tuple:
        i = self.train_size

        xenc = self.xenc[i:i+self.test_size]
        ys = self.labels[i:i+self.test_size]
        return xenc,ys
    @property
    def enc_dev_set(self)->tuple:        
        i = self.test_size+self.train_size

        xenc = self.xenc[i:i+self.dev_size]
        ys = self.labels[i:i+self.dev_size]
        return xenc, ys
    
    def get_complete_dataset(self)->tuple:
        with open(Path(__file__).parent.parent.joinpath(self.file_name), 'r') as f:
            names = f.read().splitlines()
            random.shuffle(names)
        stoi_dict = stoi()
        for name in names:
            chrs = ["."] + list(name) + ["."]
            for c1,c2,c3 in zip(chrs, chrs[1:], chrs[2:]):
                i1,i2 = stoi_dict[c1], stoi_dict[c2]
                self.xpos.append(27*i1+i2)
                self.ys.append(stoi_dict[c3])

        self.labels = torch.tensor(self.ys)
        self.xenc = F.one_hot(torch.tensor(self.xpos), num_classes=27*27).float()
        return self.xenc, self.labels