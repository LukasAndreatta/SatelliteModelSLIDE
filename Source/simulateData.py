# By Lukas Andreatta
import numpy as np

from SatelliteSlide import *
import AISurrogateLib as aiLib

calcTd = True

h    = 0.004    # step size in seconds
tEnd = 12       # end time in seconds (is equal simulation duration as time always starts at 0)
tEnd = np.ceil(tEnd/h) * h #  scale tEnd so that there is an integer value of timesteps
nTraining = 2500
nTest = int(nTraining/2)

def generationFunction(self, relCnt, isTest, dataErrorEstimator, n):
    return self.RandomPath(n)


nStepsTotal = tEnd/h
satelliteSim = SatelliteModel(nStepsTotal=nStepsTotal, endTime=tEnd,
                                generationFunction=generationFunction, generationFunctionParams=[6],
                                workspaceBounds=(4,2*np.pi), minimumTraversalTime=1.0,
                                HBCfileName='modelData/panel-abaqus.npy',
                                acceptableSolverFails=10)

nntc = aiLib.NeuralNetworkTrainingCenter(None, satelliteSim, computeDevice='cpu')
if __name__ == '__main__':
    nntc.CreateTrainingAndTestData(nTraining,nTest,
                                parameterFunction=PVCreateData, numberOfThreads=64)
    nntc.SaveTrainingAndTestsData(f'output/rawData-{nTraining}-{nTest}-{h}-{int(tEnd)}.npy')
    print(f'solver Failures: {satelliteSim.solverFailures}')

    if calcTd:
        inV = satelliteSim.CreateZeroInputVector()
        satelliteSim.SetTrajectory(inV)
        satelliteSim.CreateModel()
        
        td, ew = aiLib.getDamping(satelliteSim.mbs, 0.01, nValues=1)
        td = td.flatten()[-1]
        RecNNOutLength = int(np.ceil(td/h))
        RecNNInLength = 2 * RecNNOutLength
        RecNNOverlap = int(np.ceil(RecNNInLength-nStepsTotal/np.ceil(nStepsTotal/RecNNInLength)))
        print(f'td: {td:.4f}\nrecommended:\nInLength:  {RecNNInLength}\nOutLength: {RecNNOutLength}\nOverlap:   {RecNNOverlap}')
