import numpy as np
import sys
import os
import copy

## Hacked import from parrent dir 
_original_sys_path = sys.path.copy()
try:
    # Add parent directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from SatelliteSlide import *
    import preProcessingLib
    import AISurrogateLib as aiLib
finally:
    # Restore sys.path to its original state
    sys.path = _original_sys_path



def PVTrainingFunction(parameterSet):
    preProcessor = copy.copy(parameterSet['functionData']['preProcessor'])
    satelliteSim = parameterSet['functionData']['satelliteSim']
    inputOutputSize = parameterSet['functionData']['inputOutputSize']
    run = parameterSet['functionData']['run']
    nTraining = parameterSet['nTraining']
    nTest = parameterSet['functionData']['nTest']

    layerSize = int(parameterSet['layerSize'])
    nHidden = parameterSet['nHidden']

    trainingName = f'ls{layerSize:04d}nH{nHidden}tnt{nTraining:04d}'

    structure = ['L']
    for i in range(nHidden):
        structure += ['R', 'L']
    
    preProcessor.RefileData(nTraining, nTest)
    preProcessor.SaveTrainingAndTestsData('tntData/processed.npy')


    print(layerSize)
    nnModel = MyNeuralNetwork(inputOutputSize = inputOutputSize, # input and output size, 
                            neuralNetworkTypeName = 'FFN',
                            hiddenLayerSize = layerSize,
                            hiddenLayerStructure = structure,
                            computeDevice='cuda',
                            )
    
    nntc = aiLib.NeuralNetworkTrainingCenter(nnModel, satelliteSim, computeDevice='cuda')
    nntc.LoadTrainingAndTestsData('tntData/processed.npy')

    nntc.TrainModel(maxEpochs=1e3,
                    lossLogInterval=10,
                    testEvaluationInterval=10,
                    learningRate=0.0001,)
    
    nntc.SaveLossLog(f'{run}/output/{trainingName}_lossLog.npy')
    nntc.SaveNNModel(f'{run}/output/{trainingName}_model.pth')

    netTest = nntc.EvaluateModel(plotTrainings=[0,3,4],plotTests=[0,3,4],plotVars=['time','C000'], saveFiguresPath=f'{run}/figures/{trainingName}_')
    nntc.PlotLossLog(f'{run}/figures/{trainingName}_LossLog.pdf')
    return netTest




if __name__ == '__main__':
    inputOutputSize = [(98, 12), (49,1)]

    nStepsTotal = inputOutputSize[0][0]
    nOutSteps = inputOutputSize[1][0]

    satelliteSim = SatelliteExuModel(nStepsTotal=nStepsTotal,
                                 endTime=nStepsTotal * 0.05,
                                 NNInputLength=nStepsTotal, NNInputOverlap=None, NNOutputLength=nOutSteps, minimumTraversalTime=4, workspaceBounds=(2, 2*np.pi),
                                 inputScalingPositions=1/4, outputScaling=10, outputVectorPattern=1, acceptableSolverFails=15)
    satelliteSim.SetInputOutputSize(inputOutputSize)

    preProcessor = preProcessingLib.PreProcessor(satelliteSim)
    preProcessor.LoadTrainingAndTestsData('tntData/exp01-0.05-12s-N03.npy')

    run = 'R3'
    outFileName = f'exp01-{run}'
    functionData = {'run':run,
                    'preProcessor':preProcessor,
                    'satelliteSim':satelliteSim,
                    'inputOutputSize':inputOutputSize,
                    'nTest':1000}
    

    
    outList = ParameterVariation(PVTrainingFunction,
                                 parameters={'nHidden': [1,2,3],
                                             'layerSize':list(np.round(np.array((0.5,0.75,1,1.25,1.5)) * 98 * 12)),
                                             'nTraining': [2000,3000,4000,5000]},
                    parameterFunctionData=functionData,
                    numberOfThreads=1,
                    debugMode = False,
                    addComputationIndex = True,
                    useMultiProcessing = False,
                    showProgress = True,
                    resultsFile=outFileName+'.txt')
    
    outDict={'input':outList[0], 'output':outList[1]}
    np.save(outFileName + '.npy', outDict, allow_pickle=True)