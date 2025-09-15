import numpy as np
import sys
import os
import copy
import time

## Hacked import from parrent dir 
_original_sys_path = sys.path.copy()
try:
    # Add parent directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from Satellite_Slide import *
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

    # tsTraining = time.time()
    nntc.TrainModel(maxEpochs=1e3,
                    lossLogInterval=10,
                    testEvaluationInterval=10,
                    learningRate=0.0001,)
    trainingTime = nntc.trainingTime
    print(f'training time = {trainingTime}')

     
    nntc.SaveLossLog(f'{run}/output/{trainingName}_lossLog.npy')
    nntc.SaveNNModel(f'{run}/output/{trainingName}_model.pth')

    netTest = nntc.EvaluateModel(plotTrainings=[0,3,4],plotTests=[0,3,4],plotVars=['time','C000'], saveFiguresPath=f'{run}/figures/{trainingName}_')
    nntc.PlotLossLog(f'{run}/figures/{trainingName}_LossLog.pdf')

    datasetTest = TensorDataset(torch.tensor(preProcessor.inputsTest, dtype=nntc.floatType).to(nntc.computeDevice))
    dataloaderTest = DataLoader(datasetTest, batch_size=1)

    avgtimeACC = 0
    for input in dataloaderTest:
        input = input[0]
        ts = time.time()
        nntc.nnModel(input)
        avgtimeACC += (time.time()-ts)
    avgRunTime = avgtimeACC / (preProcessor.inputsTest.shape[0])
    print(f'avgTestRunTime = {avgRunTime}')
        
    
   
    outDict={
        'avgRunTime':avgRunTime,
        'trainingTime':trainingTime
             }
    outDict.update(netTest)

    return outDict




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
    preProcessor.LoadTrainingAndTestsData('tntData/exp02-Z-0.05-12s-N03.npy')

    run = 'R1'
    outFileName = f'exp05-{run}'
    functionData = {'run':run,
                    'preProcessor':preProcessor,
                    'satelliteSim':satelliteSim,
                    'inputOutputSize':inputOutputSize,
                    'nTest':1000}
    

    outList = ParameterVariation(PVTrainingFunction,
                    parameters={'nHidden': [1,3],
                'layerSize':list(np.round(np.array((0.5,1,1.5)) * 98 * 12)),
                'nTraining': [2000,4000],
                'cnt':[0,1,2,3,4]},
                    parameterFunctionData=functionData,
                    numberOfThreads=1,
                    debugMode = False,
                    addComputationIndex = True,
                    useMultiProcessing = False,
                    showProgress = True,
                    resultsFile=outFileName+'.txt')
    
    outDict={'input':outList[0], 'output':outList[1]}
    np.save(outFileName + '.npy', outDict, allow_pickle=True)