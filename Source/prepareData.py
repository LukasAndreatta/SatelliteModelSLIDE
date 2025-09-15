#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           prepareData
#
# Details:  Prepares simulation output data for neural network training and 
#           saves the result
#
# Author:   Lukas Andreatta
# date:     2025-09-15
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from SatelliteSlide import *
import preProcessingLib

dampingDecay = 0.01 # How much the oscillation of the smallest eigen value should decay

h    = 0.004    # step size in seconds
tEnd = 12       # end time in seconds (is equal simulation duration as time always starts at 0)
tEnd = np.ceil(tEnd/h) * h #  scale tEnd so that there is an integer value of timesteps

nStepsTotal=int(tEnd/h)
print(f'nStepsTotal = {nStepsTotal}')
satelliteSim = SatelliteModel(nStepsTotal=nStepsTotal,
                                 endTime=tEnd,
                                 NNInputLength=1228, NNInputOverlap=614, NNOutputLength=614, minimumTraversalTime=4, workspaceBounds=(2, 2*np.pi),
                                 inputScalingPositions=1/4, outputScaling=10, outputVectorPattern=1, acceptableSolverFails=15)

if __name__ == '__main__':
    preProcessor = preProcessingLib.PreProcessor(satelliteSim)
    preProcessor.LoadTrainingAndTestsData('output/rawData-2500-1250-0.004-12.npy')

    preProcessor.Resample(hNew = 0.05)
    preProcessor.ConvertToNNData(False)
    preProcessor.ChangeOutputPattern('N01-Norm', 1)
   
    preProcessor.DataSummary()
    scalings = preProcessor.SuggestScaling()
    
    mScalings = (np.array([1/4, 1/4, 1/4, 1,1,1,1,1,1,1,1,1]), 7.02)
    print(scalings)
    if np.linalg.norm(scalings[0]-mScalings[0]) > 0.05 or np.linalg.norm(scalings[1]-mScalings[1]) > 0.05:
        raise Warning('scalings are off')

    print(satelliteSim.GetInputOutputSizeNN())
    preProcessor.ConstTimeScale(*scalings)

    preProcessor.DataSummary()