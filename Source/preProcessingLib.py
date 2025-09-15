# By Lukas Andreatta
import numpy as np
from SatelliteSlide import LinearInterpolateArray

import torch
import os

# Resample an array, used to change the step size.
# tOffset   Use this if the first sample of the resampled array
# should be between the first and second sample of the data array
# hOld      The step size of data, the input array.
# hNew      The step size for the output.
# data:     NumPy array that should be resampled.
def resampleArray(tOffset, hOld, hNew, data):
    nStepsOld = data.shape[0]
    nStepsNew = int(np.floor((nStepsOld*hOld)/hNew))
    if nStepsNew * hNew + tOffset > nStepsOld * hOld: #avoid extrapolation
        nStepsNew -= 1

    resampled = []
    for i in range(nStepsNew):
        t = i * hNew + tOffset
        approxIDX = t / hOld
        lower = int(np.ceil(approxIDX))
        upper = int(np.floor(approxIDX))
        t0 = lower * hOld
        t1 = upper * hOld
        sample = LinearInterpolateArray(t, t0, t1, data[lower], data[upper])
        resampled.append(sample)

    return np.asarray(resampled)

# for calculating averages over data like simulation outputs, that is several time steps, each containing an array
# returns an array with the same shape as data[0], can be used to calculate the average value for each sensor output
# operation is a function like numpy.mean() or numpy.min(), that revives an array and returns a scalar value 
def _operationOnData(data, operation):
    op = []
    s = data.shape
    data = data.reshape(s[0], np.prod(s[1:
    ]))
    for i in range(len(data[0])):

        op.append(operation(data[:,i]))
    op = np.array(op).reshape(s[1:])
    return(op)

class PreProcessor:
    def __init__(self, simulationModel, useErrorEstimator = False):
        # self.nnModel = nnModel
        self.simulationModel = simulationModel
        try:
            self.h = self.simulationModel.endTime/self.simulationModel.nStepsTotal
        except NameError:
            raise ValueError('The Simulation Model must contain an endTime! Otherwise step size h can not be calculated')
        self.useErrorEstimator = useErrorEstimator
        if self.useErrorEstimator:
            raise NotImplementedError('Using the error estimator is not implemented')
        self.converted = False # Keeps track if the data has already been converted to NN data

        baseNames = ["Training", "Test"]
        if useErrorEstimator:
            baseNames += ["TrainingEst", "TestEst"]
        
        self.baseNames = np.array(baseNames)
        self.prefixes = ['inputs', 'targets', 'hiddenInit']
        self.fullNames = []
        for name in baseNames:
            for pre in self.prefixes:
                self.fullNames.append(f"{pre}{name}")
            


    # get stored simulation model
    def GetSimModel(self):
        return self.simulationModel
    
    ## derived from Peter Manzls AISurrogateLib
    #load data from .npy file
    #allows loading data with less training or test sets
    def LoadTrainingAndTestsData(self, fileName, nTraining=None, nTest=None, ignoreVersion=False):
        self.converted = False 

        fileExtension = ''
        if len(fileName) < 4 or fileName[-4:]!='.npy':
            fileExtension = '.npy'
            
        with open(fileName+fileExtension, 'rb') as f:
            dataDict = np.load(f, allow_pickle=True).all()   #allow_pickle=True for lists or dictionaries; .all() for dictionaries
        
        if not ignoreVersion:
            if dataDict['version'] < 3:
                raise ValueError(f"data dict version {dataDict['version']} is too old, must be at least 3")

        if nTraining == None:
            nTraining = dataDict['nTraining']
        if nTest == None:
            nTest = dataDict['nTest']

        if self.h != dataDict['hSim']:
            raise ValueError(f"data and model do not have the same step size! (Model: {self.h} vs data: {dataDict['hSim']})")
        
        if dataDict['nTraining'] < nTraining:
            raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available training sets ('+
                                str(dataDict['nTraining'])+') are less than requested: '+str(nTraining))
        if dataDict['nTest'] < nTest:
            raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available test sets ('+
                                str(dataDict['nTest'])+') are less than requested: '+str(nTest))

        self.inputsTraining = dataDict['inputsTraining'][:nTraining]
        self.targetsTraining = dataDict['targetsTraining'][:nTraining]
        self.hiddenInitTraining = dataDict['hiddenInitTraining'][:nTraining]
        self.inputsTest = dataDict['inputsTest'][:nTest]
        self.targetsTest = dataDict['targetsTest'][:nTest]
        self.hiddenInitTest = dataDict['hiddenInitTest'][:nTest]

        self.floatType = torch.float
        if self.useErrorEstimator:
            self.floatType = dataDict['floatType']
        
        if self.useErrorEstimator: 
            if not('inputsTrainingEst' in dataDict.keys()): 
                raise ValueError('LoadTrainingAndTestData failed because of missing data for Estimator. Is the data created correctly?')
            self.inputsTrainingEst = dataDict['inputsTrainingEst'][:nTraining]
            self.targetsTrainingEst = dataDict['targetsTrainingEst'][:nTraining]
            self.hiddenInitTrainingEst = dataDict['hiddenInitTrainingEst'][:nTraining]
            self.inputsTestEst = dataDict['inputsTestEst'][:nTest]
            self.targetsTestEst = dataDict['targetsTestEst'][:nTest]
            self.hiddenInitTestEst = dataDict['hiddenInitTestEst'][:nTest]
                
        #convert such that torch does not complain about initialization with lists:
        self.inputsTraining = np.stack(self.inputsTraining, axis=0)
        self.targetsTraining = np.stack(self.targetsTraining, axis=0)
        self.inputsTest = np.stack(self.inputsTest, axis=0)
        self.targetsTest = np.stack(self.targetsTest, axis=0)
        self.hiddenInitTraining = np.stack(self.hiddenInitTraining, axis=0)
        self.hiddenInitTest = np.stack(self.hiddenInitTest, axis=0)

    def SaveTrainingAndTestsData(self, fileName, scaled = True):
        fileExtension = ''
        if len(fileName) < 4 or fileName[-4:]!='.npy':
            fileExtension = '.npy'
        
        os.makedirs(os.path.dirname(fileName+fileExtension), exist_ok=True)
        
        dataDict = {}
        
        dataDict['version'] = 3 # to check correct version
        dataDict['ModelName'] = self.GetSimModel().GetModelName() # to check if correct name
        dataDict['inputShape'] = self.GetSimModel().GetInputScaling().shape # to check if correct size
        dataDict['outputShape'] = self.GetSimModel().GetOutputScaling().shape # to check if correct size
        dataDict['nTraining'] = self.targetsTraining.shape[0]
        dataDict['nTest'] = self.targetsTraining.shape[0]
        try:
            dataDict['hSim'] = self.h
        except NameError:
            raise ValueError('The Simulation Model must contain an endTime! Otherwise step size h can not be calculated')

        dataDict['inputsTraining'] = self.inputsTraining
        dataDict['targetsTraining'] = self.targetsTraining
        dataDict['inputsTest'] = self.inputsTest
        dataDict['targetsTest'] = self.targetsTest

        # initialization of hidden layers in RNN (optional)
        dataDict['hiddenInitTraining'] = self.hiddenInitTraining
        dataDict['hiddenInitTest'] = self.hiddenInitTest

        dataDict['floatType'] = self.floatType
        
        # same version 2 from here on
        if self.useErrorEstimator: 
            dataDict['inputsTrainingEst'] = self.inputsTrainingEst
            dataDict['targetsTrainingEst'] = self.targetsTrainingEst
            dataDict['inputsTestEst'] = self.inputsTestEst
            dataDict['targetsTestEst'] = self.targetsTestEst
            
            dataDict['hiddenInitTrainingEst'] = self.hiddenInitTrainingEst
            dataDict['hiddenInitTestEst'] = self.hiddenInitTestEst
        
        with open(fileName+fileExtension, 'wb') as f:
            np.save(f, dataDict, allow_pickle=True) #allow_pickle=True for lists or dictionaries

    # combines data into so that Training and Test data are in a single list.
    # if data is not None:
    # dataDict example([inputTraining, inputTest], [targetsTraining, targetsTest], [hiddenInitTraining, hiddenInittest])
    def _CompileData(self, dataDict=None , dataSelector=None):
        if dataSelector == None:
            baseNames = self.baseNames
        else:
            try:
                baseNames = self.useErrorEstimator[dataSelector]
            except (TypeError, IndexError):
                raise ValueError(f"data selector invalid must be an index between 0 and {len(self.baseNames) + 1}\n(Error Estimator is {'used'*self.useErrorEstimator}{'not used'* (not self.useErrorEstimator)})")

        simInputs = []
        simOutputs = []
        simHiddens = []
        modes = []

        attributeList = []
        if dataDict is None:
            for i, name in enumerate(baseNames):
                attributeList.append(f"inputs{name}")
                arr = getattr(self, attributeList[-1])
                length = len(arr)
                simInputs += [arr[i] for i in range(arr.shape[0])]
                attributeList.append(f"targets{name}")
                arr = getattr(self, attributeList[-1])
                simOutputs += [arr[i] for i in range(arr.shape[0])]
                attributeList.append(f"hiddenInit{name}")
                arr = getattr(self, attributeList[-1])
                simHiddens += [arr[i] for i in range(arr.shape[0])]

                modes += [i] * length
        else:
            for i, name in enumerate(baseNames):
                attributeList.append(f"inputs{name}")
                arr = dataDict[attributeList[-1]]
                length = len(arr)
                simInputs += [arr[i] for i in range(arr.shape[0])]
                
                attributeList.append(f"targets{name}")
                arr = dataDict[attributeList[-1]]
                simOutputs += [arr[i] for i in range(arr.shape[0])]
                
                attributeList.append(f"hiddenInit{name}")
                arr = dataDict[attributeList[-1]]
                simHiddens += [arr[i] for i in range(arr.shape[0])]

                modes += [i] * length


        return np.array(simInputs), np.array(simOutputs), np.array(simHiddens), np.array(modes), np.array(attributeList)

    # ad a sample to the data stored in this class
    def _AddData(self, input, target, hidden, mode):
        baseName = self.baseNames[mode]
        try:
            if input == None:
                skipInput = True
            else:
                skipInput = False
        except:
             skipInput = False
        if not skipInput:
            inputs = list(getattr(self, f'inputs{baseName}'))
            setattr(self, f'inputs{baseName}', np.array(inputs + list(input)))
        
        try:
            if target == None:
                skipTarget = True
            else:
                skipTarget = False
        except:
             skipTarget = False
        if not skipTarget:
            targets = list(getattr(self, f'targets{baseName}'))
            setattr(self, f'targets{baseName}', np.array(targets + list(target)))

        try:
            if hidden == None:
                skipHidden = True
            else:
                skipHidden = False
        except:
             skipHidden = False
        if not skipHidden:
            hiddenInit = list(getattr(self, f'hiddenInit{baseName}'))
            setattr(self, f'hiddenInit{baseName}', np.array(hiddenInit + list(hidden)))

    # Convert from simulation output data (inputs the same length as outputs) to
    # SLIDE NN data, where the targets are shorter than the inputs.
    def ConvertToNNData(self, useModelScaling, data=None):
        if self.converted and not data is None:
            print('PreProcessor: data was already converted to nn data')
            return
        
        simInputs, simOutputs, simHiddens, modes, attributeList = self._CompileData(data)

        if data is None:
            for attr in attributeList:
                setattr(self, attr, np.array([]))

        for simInput, simOutput, simHidden, mode in list(zip(simInputs, simOutputs, simHiddens, modes)):
            values = [self.simulationModel.ConvertToNNInputOutput(simInput, simOutput, simHidden, useModelScaling)]
            for i, item in enumerate(values):
                self._AddData(*item, mode)
            
        for attr in attributeList:
            arr = getattr(self, attr)
            setattr(self, attr, np.array(arr))
        self.converted = True

    # Load additional data from a file, data must be converted beforehand.
    def LoadAdditional(self, fileName, useModelSacaling = False, newDataAlreadyConverted=False):
        fileExtension = ''
        if len(fileName) < 4 or fileName[-4:]!='.npy':
            fileExtension = '.npy'
            
        with open(fileName+fileExtension, 'rb') as f:
            dataDict = np.load(f, allow_pickle=True).all()
        
        if dataDict['version'] < 3:
            raise ValueError(f"data dict version {dataDict['version']} is too old, must be at least 3")
        
        
        nTraining = dataDict['nTraining']
        nTest = dataDict['nTest']

        if self.h != dataDict['hSim']:
            raise ValueError(f"data and model do not have the same step size! (Model: {self.h} vs data: {dataDict['hSim']})")


        if dataDict['nTraining'] < nTraining:
            raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available training sets ('+
                                str(dataDict['nTraining'])+') are less than requested: '+str(nTraining))
        if dataDict['nTest'] < nTest:
            raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available test sets ('+
                                str(dataDict['nTest'])+') are less than requested: '+str(nTest))


        if (not self.converted) or newDataAlreadyConverted:
            if self.inputsTraining[0].size != dataDict['inputsTraining'][0].size or self.inputsTraining[0].shape != dataDict['inputsTraining'][0].shape:
                raise ValueError('inputs do not match up (size or shape)')
            if self.targetsTraining[0].size != dataDict['targetsTraining'][0].size or self.targetsTraining[0].shape != dataDict['targetsTraining'][0].shape:
                raise ValueError('targets do not match up (size or shape)')
            if useModelSacaling == True:
                raise RuntimeError('model scaling can only be used if the data that was loaded before is already converted to nnData')

            self.inputsTraining = np.vstack((self.inputsTraining, dataDict['inputsTraining'][:nTraining]))
            self.targetsTraining = np.vstack((self.targetsTraining, dataDict['targetsTraining'][:nTraining]))
            self.hiddenInitTraining = np.vstack((self.hiddenInitTraining, dataDict['hiddenInitTraining'][:nTraining]))

            self.inputsTest = np.vstack((self.inputsTest, dataDict['inputsTest'][:nTest]))
            self.targetsTest = np.vstack((self.targetsTest, dataDict['targetsTest'][:nTest]))
            self.hiddenInitTest = np.vstack((self.hiddenInitTest, dataDict['hiddenInitTest'][:nTest]))

        elif self.converted and (not newDataAlreadyConverted) :
            self.ConvertToNNData(useModelSacaling, dataDict)
        else:
            raise RuntimeError()

        self.floatType = torch.float
        if self.useErrorEstimator:
            self.floatType = dataDict['floatType']
        
        if self.useErrorEstimator: 
            raise NotImplemented()

    # Change the number of test and training data in the data set,
    # nTraining and nTest, the desired number of test and training samples.
    # The first nTraing samples are the training data.
    # The last nTest samples are the test data.
    # If nTest is None, all samples that are not training 
    # samples will be test samples.
    def RefileData(self, nTraining, nTest=None):
        inputs, targets, hiddens, modes, attributeList = self._CompileData()
        nAvailable = inputs.shape[0]
        if nTest is None:
            nTest = nAvailable-nTraining
        
        if nTest < 1:
            raise ValueError('at least one test is required')
        
        if nTraining+nTest > nAvailable:
            raise ValueError('not enough data available')
        
        
        self.inputsTraining =  inputs[0:nTraining]
        self.targetsTraining =  targets[0:nTraining]
        self.hiddenInitTraining =  hiddens[0:nTraining]

        self.inputsTest=  inputs[-nTest:]
        self.targetsTest =  targets[-nTest:]
        self.hiddenInitTest =  hiddens[-nTest:]
        

    def DataSummary(self):
        inputs, targets, hiddens, modes, attributeList = self._CompileData()
        dataList =  [inputs, targets]
        print(f'\nnTraining = {self.inputsTraining.shape[0]}\nnTest = {self.inputsTest.shape[0]}')
        print(f'nTotal = {inputs.shape[0]}\n')

    # Used to print various properties of the data,
    # adapt for what is interesting for the specific purpose
    # dataSelector: What should be Inspected
    #       None: all data
    #       0:    Training Data
    #       1:    Test Data
    #       2:    Training Data Error estimator
    #       3:    Test Data Error estimator
    def InspectData(self, dataSelector=None):
        if self.simulationModel.outputVectorPattern != 1:
            raise RuntimeError('outputVectorPattern has to be 1 to use inspectData')
        inputs, targets, hiddens, modes, attributeList = self._CompileData(dataSelector)
        dataList =  [inputs, targets]
        print(f'nInput = {inputs.shape[0]}\nnTargets = {targets.shape[0]}')

        print(f'Averages:')
        for data in dataList:
            opData = []
            for j, vec in enumerate(data):
                op = _operationOnData(vec, np.mean)
                opData.append(op)
            opData = np.array(opData)
            print(_operationOnData(opData, np.mean))

        inputs, targets, hiddens, modes, attributeList = self._CompileData(dataSelector)
        inMax = np.max(abs(inputs), axis=(0, 1))
        outMax = np.max(abs(targets), axis=(0, 1))
        print(inMax)
        print()
        print(outMax)
    
    # Plot the Input data
    def PlotInputs(self, toplot):
        import matplotlib.pyplot as plt

        for i in toplot:
            
            plt.plot(self.inputsTraining[i][:,0], label=f'in {i}')
            if len(self.targetsTraining[i][:].shape) == 1:
                plt.plot(self.targetsTraining[i][:], label=f'out {i}')
            else:
                plt.plot(self.targetsTraining[i][:,0], label=f'out {i}')
        plt.grid(True)
        plt.savefig('out/figures/inpuPlot.pdf')

    # Suggest Scaling values, change as needed for the specific purpose.
    # Scaling by multiplying with the inverse of the maximum keeping the data between -1 and 1    
    def SuggestScaling(self, dataSelector=None):
        inputs, targets, hiddens, modes, attributeList = self._CompileData(dataSelector)
        inMax = np.max(abs(inputs), axis=(0, 1))
        outMax = np.max(abs(targets), axis=(0, 1))
        return 1/inMax, 1/outMax
    
    # scale the vector with the same factor regardless of time
    def ConstTimeScale(self, inFactors, outFactors):
        if self.simulationModel.GetInputScaling()[0].shape != inFactors.shape:
            raise ValueError(f'inFactors are the wrong shape ({inFactors.shape} instead of {self.simulationModel.GetInputScaling()[0].shape})')
        if self.simulationModel.GetOutputScaling()[0].shape != outFactors.shape:
            raise ValueError(f'outFactors are the wrong shape ({outFactors.shape} instead of {self.simulationModel.GetOutputScaling()[0].shape})')
        inScaling = np.broadcast_to(inFactors, (self.inputsTraining[0].shape[0], *inFactors.shape))
        outScaling = np.broadcast_to(outFactors, (self.targetsTraining[0].shape[0], *outFactors.shape))
        scalings = [inScaling, outScaling]

        for baseName in self.baseNames:
            for i, prefix in enumerate(['inputs', 'targets']):
                vec = getattr(self, prefix+baseName)
                setattr(self, prefix+baseName, vec*scalings[i])
                
    # Resample, change the step size of the data
    def Resample(self, hNew):
        if self.converted:
            raise NotImplementedError()

        hOld = self.h
        if hNew < hOld:
            raise ValueError('can not upsample')

        inputs, targets, hiddens, modes, attributeList = self._CompileData()

        inputs = list(inputs)
        targets = list(targets)
        for i, inputVec in enumerate(inputs):
            inputs[i] = resampleArray(0, hOld, hNew, inputVec)
        for i, targetVec in enumerate(targets):
            targets[i] = resampleArray(0, hOld, hNew, targetVec)
        
        nnInSteps = self.simulationModel.NNInputLength
        nnOutSteps = self.simulationModel.NNOutputLength
        self.simulationModel.NNInputOverlap = int(np.floor(self.simulationModel.NNInputOverlap * hOld/hNew))

        nnInStepsNew = int(np.floor(nnInSteps * hOld/hNew))
        nnOutStepsNew = int(np.floor(nnOutSteps * hOld/hNew))
        
        for name in attributeList:
            setattr(self, name, np.array([]))
        for inputVec, targetVec, hidden, mode in zip(inputs, targets, hiddens, modes):
            self._AddData([inputVec], [targetVec], [hidden], mode)
        self.h = hNew
        inVecPattern = self.simulationModel.inputVectorPattern
        outVecPattern = self.simulationModel.outputVectorPattern
        self.simulationModel.SetInputScaling(self.simulationModel.GetInputScaling()[0:nnInStepsNew])
        self.simulationModel.SetOutputScaling(self.simulationModel.GetOutputScaling()[0:nnOutStepsNew])
        self.simulationModel.NNInputLength = nnInStepsNew
        self.simulationModel.NNOutputLength = nnOutStepsNew
        self.simulationModel.inputVectorPattern = inVecPattern
        self.simulationModel.outputVectorPattern = outVecPattern
        self.simulationModel.nStepsTotal = nnInStepsNew
        self.simulationModel.endTime = nnInStepsNew * hNew

    # Change the output pattern, adapt for the specific purpose.
    def ChangeOutputPattern(self, outputVectorPattern, outputScalingFactor = 1):
        if self.simulationModel.outputVectorPattern != 1:
            raise RuntimeError('outputVectorPattern has to be 1 to use ChangeOutputPattern')
        inputs, targets, hiddens, modes, attributeList = self._CompileData()
        for name in self.baseNames:
            setattr(self, f'targets{name}', np.array([]))

        if outputVectorPattern.lower() == 'N01-Norm'.lower(): #1 Sensor (One of the outermost sensors) and the norm of the deformation
            for target, mode in zip(targets, modes):
                target = np.linalg.norm(target[:,4], axis=1)
                target = target.reshape(target.size, 1)
                self._AddData(None, np.array([target]), None, mode)
            outputScalingVector = np.ones(self.targetsTraining[0].shape) * outputScalingFactor
            self.simulationModel.SetOutputScaling(outputScalingVector, outputScalingFactor = 1)

        elif outputVectorPattern.lower() == 'N01-Z'.lower(): #1 Sensor (One of the outermost sensors) Z
            for target, mode in zip(targets, modes):
                target = target[:,4,2]
                target = target.reshape(target.size, 1)
                self._AddData(None, np.array([target]), None, mode)
            outputScalingVector  = np.ones(self.targetsTraining[0].shape) * outputScalingFactor
            self.simulationModel.SetOutputScaling(outputScalingVector, outputScalingFactor = 1)
        elif outputVectorPattern.lower() == 'N01-3D'.lower(): #1 Sensor 3D deformation
            for target, mode in zip(targets, modes):
                target = target[:, 4]
                self._AddData(None, np.array([target]), None, mode)

            outputScalingVector  = np.ones(self.targetsTraining[0].shape) * outputScalingFactor
            self.simulationModel.SetOutputScaling(outputScalingVector, outputScalingFactor = 1)

        elif outputVectorPattern.lower() == 'N04-Norm'.lower(): #4 outermost sensors and the norm of the deformation
            for target, mode in zip(targets, modes):
                target = np.linalg.norm(target[:, [4, 5, 10, 11]], axis=(2))
                self._AddData(None, np.array([target]), None, mode)

            outputScalingVector  = np.ones(self.targetsTraining[0].shape) * outputScalingFactor
            self.simulationModel.SetOutputScaling(outputScalingVector, outputScalingFactor = 1)

        elif outputVectorPattern.lower() == 'N04-Z'.lower(): #4 outermost sensors Z
            for target, mode in zip(targets, modes):
                target = target[:, [4, 5, 10, 11], 2]
                self._AddData(None, np.array([target]), None, mode)

            outputScalingVector  = np.ones(self.targetsTraining[0].shape) * outputScalingFactor
            self.simulationModel.SetOutputScaling(outputScalingVector, outputScalingFactor = 1)

        elif outputVectorPattern.lower() == 'N04-3D'.lower(): #4 outermost sensors 3D deformation
            for target, mode in zip(targets, modes):
                target = target[:, [4, 5, 10, 11]]
                self._AddData(None, np.array([target]), None, mode)

            outputScalingVector  = np.ones(self.targetsTraining[0].shape) * outputScalingFactor
            self.simulationModel.SetOutputScaling(outputScalingVector, outputScalingFactor = 1)

        else:
            raise ValueError(f'invalid output vector pattern: {outputVectorPattern}')