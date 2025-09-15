# adapted from:
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           04_Flexible_6RRobot
# Details:  File for creating data and learning the the socket's deflection of 
#           a robot standing on the flexible socket. 
#
# Author:   Peter Manzl, Johannes Gerstmayr
# Date:     2024-10-01
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# adapted by Lukas Andreatta
import torch 
from torch import nn
import numpy as np
import os
import sys
_original_sys_path = sys.path.copy()
try:
    # Add parent directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from AISurrogateLib import *
finally:
    # Restore sys.path to its original state
    sys.path = _original_sys_path

class CustomNN(MyNeuralNetwork):
    def __init__(self, inputOutputSize = None, hiddenLayerSize = None, hiddenLayerStructure = ['L'], 
                 neuralNetworkTypeName = 'RNN', computeDevice = 'cpu', 
                 activationType = 0, typeNumber = 0, #'relu', None; 'tanh' is the default
                 ):
        super().__init__(inputOutputSize = inputOutputSize, hiddenLayerSize = hiddenLayerSize, hiddenLayerStructure = hiddenLayerStructure, 
                      neuralNetworkTypeName = neuralNetworkTypeName, computeDevice = computeDevice,
                      )
        self.myNN = self.customNetworks(inputOutputSize, hiddenLayerSize, typeNumber = typeNumber, activationType=activationType, computeDevice=computeDevice)
    
    # create cusotom networks depending on activationType and typenumber
    def customNetworks(self, inputOutputSize, hiddenLayerSize, typeNumber = 0, activationType = 0, computeDevice = 'cpu'): 
        activationType  = GetActivationType()[activationType]

        self.typeNumber  = typeNumber 
        nnList = []        
        

        # this part of the model is the same for everybody!
        nInTotal = np.prod(inputOutputSize[0])
        nOutTotal = np.prod(inputOutputSize[1])
        nOutSplit = nOutTotal //3 
        self.nInTotal, self.nOutTotal, self.nOutSplit = nInTotal, nOutTotal, nOutSplit
        model = nn.ModuleList([nn.Flatten(1), 
                                # nn.Linear(nInTotal, nInTotal)
                                ])
        
        # modelSplit = []
        for i in range(3): # 3 outputs: x,y,z! 
            # activationType = nn.ELU # fixed to ELU
            activationType = nn.ReLU
            modelSplit = nn.Sequential(nn.Linear(nInTotal, hiddenLayerSize))
            modelSplit.append(activationType())
            modelSplit.append(nn.Linear(hiddenLayerSize, nOutSplit))
            modelSplit.to(computeDevice)
            model.append(modelSplit)

        model.append(nn.Unflatten(-1, inputOutputSize[1]))
        model.to(computeDevice)
        self.xOut = None
            
            
        print('custom network: \n{}'.format(model))
        return model
    
    def forward(self, x): 
    
        xIn_split = self.myNN[0](x)
        if self.xOut is None or self.xOut.shape !=[x.shape[0],self.nOutTotal]:     
            self.xOut = torch.zeros([x.shape[0],self.nOutTotal]).to(self.computeDevice)
        iOffset = 0
        for i in range(3): 
            xOut_split = self.myNN[i+1](xIn_split)
            self.xOut[:,iOffset:iOffset+self.nOutSplit] = xOut_split
            iOffset += self.nOutSplit
        return self.myNN[-1](self.xOut)