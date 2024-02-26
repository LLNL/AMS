import torch
import os
import sys
import numpy as np
from torch.autograd import Variable
from torch import jit

class ConstantModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize, constant):
        super(ConstantModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.linear.weight.data.fill_(0.0)
        self.linear.bias.data.fill_(constant)

    def forward(self, x):
        y = self.linear(x)
        return y

def main(args):
    inputDim = int(args[1])
    outputDim = int(args[2])
    device = args[3]
    enable_cuda = True
    if device == "cuda":
        enable_cuda = True
        suffix = '_gpu'
    elif device == "cpu":
        enable_cuda = False
        suffix = '_cpu'
    
    model = ConstantModel(inputDim, outputDim, 1.0).double()
    if torch.cuda.is_available() and enable_cuda:
        model = model.cuda()

    model.eval()
    with torch.jit.optimized_execution(True):
        traced = torch.jit.trace(model, (torch.randn(inputDim, dtype=torch.double), ))
        traced.save(f"ConstantOneModel_{suffix}.pt")

    model = ConstantModel(inputDim, outputDim, 0.0).double()
    if torch.cuda.is_available() and enable_cuda:
        model = model.cuda()

    model.eval()
    with torch.jit.optimized_execution(True):
        traced = torch.jit.trace(model, (torch.randn(inputDim, dtype=torch.double), ))
        traced.save(f"ConstantZeroModel_{suffix}.pt")

    inputs = Variable(torch.from_numpy(np.zeros((1, inputDim))))
    zero_model = jit.load(f"ConstantZeroModel_{suffix}.pt")
    print("ZeroModel", zero_model(inputs))

    one_model = jit.load(f"ConstantOneModel_{suffix}.pt")
    print("OneModel", one_model(inputs))




if __name__ == '__main__':
    main(sys.argv)




