# Based on simModels.py from: \url{https://github.com/peter-manzl/SLIDE}
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           SatelliteSlide
#
# Details:  Class used to implement the satellite model in the SLIDE framework
#
# Author:   Lukas Andreatta
# date:     2025-09-15
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import exudyn as exu
from exudyn.utilities import *
from exudyn.FEM import *

from simModels import SimulationModel, ModelComputationType
from AISurrogateLib import * 

import warnings
import numpy as np

import TrajectoryFunctions as tf
from SatilliteExudynModel import createSatellite



def isInt(val):
    try:
        val = float(val)
        return val.is_integer()
    except (ValueError, TypeError):
        return False
    

class SatelliteModel(SimulationModel):
    positions = [] # Trajectory positions
    rotations = [] # Trajectory rotation matrices

    # nStepsTotal:          total steps for the Input- and Output vector
    # endTime:              determines how long the Simulation will run
    # NNInputLength:        int, how many steps the neural networks input vector should have (must be shorter then nStepsTotal)
    # NNOutputLength:       int, how many steps the neural networks output vector should have (must be shorter then nStepsTotal and NNInputLength)
    # NNInputOverlap: int, how many steps of one input vector overlap with the next input vector (must be less NNInputLength and should not be negative. It can be 0.)
    #
    # trajectorySource:     if trajectories are loaded or generated
    #       0: generate trajectories
    #       1: load from file
    #
    # generationFunction:   function that will be used by CreateInputVector to generate Trajectories:
    #       function will be called with:
    #       generationFunction(self, relCnt, isTest, dataErrorEstimator, *generationFunctionParams)
    #       parameters are the same as the ones CreateInputVector plus params which is a tuple containing additional parameters
    #       if None a default generator will be used
    #
    # generationFunctionParams: see generationFunction can be None
    #
    # workspaceBounds:      only needed for trajectory generation: tuple(float,float)
    #       first float: defines the size of a box (in m) around the origin. The box  will have a side length of twice that value
    #       second float: defines the maximum absolute rotation around any of the coordinate axes.
    #
    # minimumTraversalTime: float (seconds) used as a minimum time between to poses in random trajectory generation. The system should be able to physically transition between the workspace bounds in that time. That is go between the most extreme values allowed by workspaceBounds.
    # acceptableSolverFails: how often the solver is allowed to fail, a new trajectory will be generated after the solver fails, until that number is reached
    #
    # inputVectorPattern:   describes the Pattern/Shape of the input vector:
    #       1: np.array of length nStepsTotal containing np.arrays of length 12: the first 3 entries are The position of the flange, the other 9 entries are the flattened rotation Matrix of the flange.
    #
    # outputVectorPattern:  describes the Pattern/Shape of the output vector:
    #       0: very simple output: one Deformation in one Direction
    #       1: np.array of length nStepsTotal containing np.arrays of length :
    #      -1: unknown, custom scaling, used for preprocessing
    # inputScalingPositions:input scaling factors for the positions (no scaling is needed for rotations)
    # outputScaling:        output scaling factor for the deformations
    # HBCfileName:          filename to load the HBC modes
    # femInterface:         import a fem interface directly, HBCfileName will be ignored    

    def __init__(self, nStepsTotal=100,
                 endTime=1,
                 NNInputLength=None, NNOutputLength=None, NNInputOverlap = None,
                 trajectorySource=0,
                 generationFunction = None, generationFunctionParams = None,
                 workspaceBounds=(1,2*np.pi), minimumTraversalTime=2.0,
                 acceptableSolverFails = 0,
                 inputVectorPattern = 1, outputVectorPattern = 1,
                 inputScalingPositions = 1.0, outputScaling = 1.0,
                 HBCfileName = 'modelData/panel-abaqus.npy',
                 femInterface = None
                 ):

        SimulationModel.__init__(self)

        if not isInt(nStepsTotal):
            raise ValueError('nStepsTotal must be an integer value')
        nStepsTotal = int(nStepsTotal)
        self.nStepsTotal = nStepsTotal
        self.endTime = endTime

        if NNInputLength == None:
            NNInputLength = nStepsTotal
        if NNOutputLength == None:
            NNOutputLength = NNInputLength
        if NNInputOverlap == None:
            NNInputOverlap = 0

        if NNInputLength > nStepsTotal:
            raise ValueError('NNInputLength must not be grater than nStepsTotal')
        if NNOutputLength > NNInputLength:
            raise ValueError('NNOutputLength must not be grater than NNInputLength')
        if NNInputOverlap >= NNInputLength:
            raise ValueError('NNInputOverlap must not be smaller than NNInputLength')

        self.NNInputLength = NNInputLength
        self.NNOutputLength = NNOutputLength
        self.NNInputOverlap = NNInputOverlap

        # Trajectory Generation Setup:
        self.trajectorySource = trajectorySource
        if trajectorySource == 0:
            self.generationFunction = generationFunction
            self.generationFunctionParams = generationFunctionParams

            # ensure length‑2 sequence
            if not hasattr(workspaceBounds, "__iter__") or len(workspaceBounds) != 2:
                raise ValueError("workspaceBounds must be a sequence of length 2")
            try:
                self.workspaceBounds = tuple(map(float, workspaceBounds))
            except (TypeError, ValueError):
                raise ValueError("workspaceBounds values must be numeric or float‑convertible")
            try:
                self.minimumTraversalTime = float(minimumTraversalTime)
                if self.minimumTraversalTime < 0:
                    raise ValueError("minimumTraversalTime must be a positive number")
            except (TypeError, ValueError):
                raise ValueError("minimumTraversalTime must be numeric or float‑convertible")
        elif trajectorySource == 1:
            raise NotImplementedError('trajectorySource == 1 is not yet implemented')
        else:
            raise ValueError(f'invalid value for trajectorySource ({trajectorySource})')
        ########


        # load HBC file.
        if femInterface == None:
            self.fem = FEMinterface()
            try:
                self.fem.LoadFromFile(HBCfileName)
                # self.cms = ObjectFFRFreducedOrderInterface(self.fem)
            except Exception as e:
                warnings.warn(f'HBC file ({HBCfileName} not found.')
        else:
            self.fem = femInterface
        ########

        # Setup the Input Vector:
        self.inputVectorPattern = inputVectorPattern
        if self.inputVectorPattern == 1:
            self.inputScaling = np.ones((self.NNInputLength, 12))
            self.inputScaling[:,0:3] = self.inputScaling[:,0:3] * inputScalingPositions
        else:
            raise ValueError(f'inputVectorPattern ({self.inputVectorPattern}) is not valid, select another value.')
        ########

        # Setup the output Vector:
        self.preProcessingMode = False
        self.outputVectorPattern = outputVectorPattern
        if self.outputVectorPattern == 1:
            self.outputScaling = np.ones((self.NNOutputLength, 12, 3)) * outputScaling
        elif self.outputVectorPattern == 0:
            self.outputScaling = np.ones((self.NNOutputLength)) * outputScaling
        elif self.outputVectorPattern == -1:
            self.preProcessingMode = True
        else:
            raise ValueError(f'outputVectorPattern ({self.outputVectorPattern}) is not valid, select another value.')
        ########
            
        self.nODE2 = 1 #always 1, as only one is measured / actuated      
        
        self.computationType = ModelComputationType.dynamicImplicit
        self.inputStep = False #only constant step functions as input
        self.trajectory = None

        self.inputScalingFactor = 1. 
        self.outputScalingFactor = 1.
        self.acceptableSolverFails = acceptableSolverFails
        self.solverFailures = 0
    
    def GetOutputXAxisVector(self):
        return np.arange(0,self.NNOutputLength)/self.nStepsTotal*self.endTime
    
    def SetInputScaling(self, inputScalingVector, inputScalingFactor = 1):
        self.inputScaling = inputScalingVector
        self.inputScalingFactor = inputScalingFactor
        self.inputVectorPattern = -1
        self.preProcessingMode = True
        
    def SetOutputScaling(self, outputScalingVector, outputScalingFactor = 1):
        self.outputScaling = outputScalingVector
        self.outputScalingFactor = outputScalingFactor
        self.outputVectorPattern = -1
        self.preProcessingMode = True

    def SetInputOutputSize(self, IOSize):
        self.SetInputScaling(np.ones(IOSize[0]), 1)

        self.SetOutputScaling(np.ones(IOSize[1]), 1)
    
    # to find the optimal input and output scaling. 
    def findScaling(self):
        inVec = self.ExtremeTest()
        outVec = self.ComputeModel(inVec)

        # Rotation matrix entries are between 0 and 1 by default
        print(f'in pos max = {max(max(abs(inVec[:,0])),max(abs(inVec[:,1])),max(abs(inVec[:,2])))}')
        print(f'out max    = {max(abs(outVec.flatten()))}')
    
 
    # fuses self.positions into the Simulation Input Vector
    # positions must be of shape (nStepsTotal, 3)
    # rotations must be a list containing 3x3 rotation matrices
    def FuseInputData(self, positions, rotations):
        trajectoryLength = len(positions)
        if trajectoryLength != len(rotations) or \
            trajectoryLength != self.nStepsTotal:
            raise ValueError('Trajectory must have nStepsTotal Length')
        
        vec = np.zeros((self.nStepsTotal, 12))
       
        for i in range(trajectoryLength):
            vec[i] = np.append(positions[i], rotations[i].flatten())
        
        return vec

    # split input data into initial values, forces or other inputs
    # return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        dataDict = {}
        raise Exception('is this ever used (delete this if you ever see it)')

        if self.inputVectorPattern == 1:
            entryLength = 12
            data = np.array(self.GetInputScaling() * inputData)

            if data.shape[1] != entryLength:
                raise ValueError(f"Shape of input Vector is invalid! ({data.shape})")

            dataDict['positions'] = []
            dataDict['rotations'] = []
            for d in data:
                dataDict['positions'].append(np.array(d[0:3]))
                dataDict['rotations'].append((np.array(d[3:12])).reshape(3,3))
        else:
            raise ValueError(f'Invalid input Vector Pattern ({self.InputVectorPattern})')
                    
        return dataDict
    
    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        dataDict = {}
        if self.outputVectorPattern == 1:
            data = np.array(outputData)
            for i in range(12):
                dataDict[f'S{i:02}X'] = data[:,i,0]
                dataDict[f'S{i:02}Y'] = data[:,i,1]
                dataDict[f'S{i:02}Z'] = data[:,i,2]
                            
        elif self.outputVectorPattern == 0:
            data = np.array(outputData)
            dataDict[f'sensorData'] = data
        elif self.outputVectorPattern == - 1:
            data = np.array(outputData)
            s = data.shape
            sl = int(np.prod(s[1:]))
            if len(s) > 1:
                data = data.reshape(s[0], sl)
                for i in range(sl):
                    dataDict[f'C{i:03}'] = np.array(data[:,i])
            else:
                dataDict[f'C000'] = data

        return dataDict
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        timeVec = self.GetOutputXAxisVector()
        dataDict = self.SplitOutputData(outputData)
        
        data = np.array(timeVec)
        for key in dataDict.keys():
            data = np.vstack((data, dataDict[key]))
        return data.T

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0}
        dataDict = self.SplitOutputData(self.GetOutputScaling()) #use GetOutputScaling as dummy data
        i = 1
        for key in dataDict.keys():
            d[key] = i
            i += 1
        return d
    
    # returns a random Point in the workspace
    def randomPoint(self):
        return np.random.rand(3) * self.workspaceBounds[0]
    
    # returns a random Rotation Matrix in the workspace
    # axes defines the axes valid values are: 'x' 'y' 'z' 'all' None, None is equal too all
    # if matrix == True then the output will be converted to a matrix
    def randomRotation(self, axes=None, matrix=True):
        if axes == None or axes == 'all':
            angles = np.random.rand(3) *  self.workspaceBounds[1]
        else:
            angle = np.random.rand() * self.workspaceBounds[1]

            if axes == 'x':
                angles = [angle, 0, 0]
            elif axes == 'y':
                angles = [0, angle, 0]
            elif axes == 'z':
                angles = [0, 0, angle]
            else:
                raise ValueError(f'Invalid Value for axes ({axes})')
        if matrix:
            return RotXYZ2RotationMatrix(angles)
        else:
            return angles

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        if self.generationFunction == None:
            return self.RandomPath(5)
        else:
            if self.generationFunctionParams == None:
                return self.generationFunction(self, relCnt, isTest, dataErrorEstimator)
            else:
                return self.generationFunction(self, relCnt, isTest, dataErrorEstimator, *self.generationFunctionParams)
        
        

    #create an input vector filled with zeros
    def CreateZeroInputVector(self):
        # preliminary trajectory with points and euler angles in one vector for each timestep, still has to be converted to positions and rotations for FuseInputData
        poseTrajectory = np.zeros((self.nStepsTotal, 6))

        # convert into positions and rotations for FuseInputData
        positions = np.zeros((self.nStepsTotal, 3))
        rotations = np.zeros((self.nStepsTotal, 3, 3))

        timeSteps = np.linspace(0, self.endTime, self.nStepsTotal)
        for i in range(len(timeSteps)):
            positions[i] = poseTrajectory[i,0:3]
            rotations[i] = RotXYZ2RotationMatrix(poseTrajectory[i,3:6])

        return self.FuseInputData(positions, rotations)

    # creates a random path with a maximum of nPoints, fixed points
    def RandomPath(self, nPoints, interpolationFunction = tf.parabolicSmooth, verbose = False, customEndTime = None):
        poseList = [np.hstack((self.randomPoint().flatten(), self.randomRotation(matrix=False)))]

        if IsNone(customEndTime):
            endTime = self.endTime
        else:
            endTime = customEndTime


        for i in range(1, nPoints):
            # rotation still in as euler angles in radians 
            poseList.append(np.hstack((self.randomPoint().flatten(), self.randomRotation(matrix=False))))
        

        # preliminary trajectory with points and euler angles in one vector for each timestep, still has to be converted to positions and rotations for FuseInputData
        poseTrajectory = np.zeros((self.nStepsTotal, 6))

        timeSteps = np.linspace(0, endTime, self.nStepsTotal)
        for j in range(6):
            pointXVals = [0]

            for i in range(1, nPoints):
                pointXVals.append(pointXVals[i-1] + self.minimumTraversalTime + np.random.rand() * (max(endTime - pointXVals[i-1], self.minimumTraversalTime)))

            for i in range(len(timeSteps)):
                t = timeSteps[i]
                poseTrajectory[i,j] = tf.multiPointTrajectory(t, pointXVals, poseList, interpolationFunction)[j]

        # convert into positions and rotations for FuseInputData
        positions = np.zeros((self.nStepsTotal, 3))
        rotations = np.zeros((self.nStepsTotal, 3, 3))

        for i in range(len(timeSteps)):
            positions[i] = poseTrajectory[i,0:3]
            rotations[i] = RotXYZ2RotationMatrix(poseTrajectory[i,3:6])

        if verbose:
            print(f'trajectory Points:\n{poseList}\n\ntimes example:{poseList}\n')

        return self.FuseInputData(positions, rotations)


    # Test a very Extreme trajectory to see if the results make sense. Ideally this would be the 'worst' trajectory the Random Path could come up with. 
    def ExtremeTest(self, interpolationFunction = tf.parabolicSmooth):      
        poseList = []
        nPoints = int(np.floor(self.endTime/self.minimumTraversalTime))
        tm, rm = self.workspaceBounds
        poseList.append(np.array((tm,tm,tm,rm,rm,rm)))
        for i in range(0, nPoints):
            poseList.append(-poseList[-1])

        # preliminary trajectory with points and euler angles in one vector for each timestep, still has to be converted to positions and rotations for FuseInputData
        poseTrajectory = np.zeros((self.nStepsTotal, 6))

        timeSteps = np.linspace(0, self.endTime, self.nStepsTotal)
        pointXVals = [0]
        for i in range(1, nPoints + 1):
            pointXVals.append(i * self.minimumTraversalTime)
        for j in range(6):
            for i in range(len(timeSteps)):
                t = timeSteps[i]
                poseTrajectory[i,j] = tf.multiPointTrajectory(t, pointXVals, poseList, interpolationFunction)[j]

        # convert into positions and rotations for FuseInputData
        positions = np.zeros((self.nStepsTotal, 3))
        rotations = np.zeros((self.nStepsTotal, 3, 3))

        for i in range(len(timeSteps)):
            positions[i] = poseTrajectory[i,0:3]
            rotations[i] = RotXYZ2RotationMatrix(poseTrajectory[i,3:6])

        return self.FuseInputData(positions, rotations)

    # input data will be written into the internal trajectory (self.positions and self.rotations) which can be used by the prestep function 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False, cnt=None):
        if self.preProcessingMode:
            raise RuntimeError('Compute Model is not allowed in Preprocessing mode')
        if hiddenData != None:
            try:
                if len(hiddenData) != 0:
                    raise NotImplementedError('Hidden data was provided, hidden data is not necessary or implemented for this model.')
            except TypeError:
                raise TypeError()

        try:
            if inputData is None:
                raise ValueError("No input data provided.")
            
            inputData = np.asarray(inputData)
            if np.any(inputData == None) or np.any(np.isnan(inputData.astype(float))):
                raise ValueError("Input data contains missing or NaN values.")
            self.SetTrajectory(inputData)

        except (TypeError, ValueError) as e:
            raise ValueError(f"Input data is not valid: {e}")
      
        

        solved = False
        while not solved:
            try:
                start = time.time()
                self.CreateModel()
                self.simulationSettings.timeIntegration.verboseMode = verboseMode
                self.mbs.SolveDynamic(self.simulationSettings)
                print(f'solution found in {time.time()-start:.2}s')
                solved = True
            except ValueError:
                self.solverFailures += 1
                if  self.solverFailures < self.acceptableSolverFails:                    
                    print(f'Failed\nold npositions[0]: {self.positions[0]}\n\n')
                    inputData = self.CreateInputVector()
                    self.SetTrajectory(np.asarray(inputData))
                    if cnt != None:
                        print(f'cnt: {cnt}')
                    print(f'!!!Solver Fail Count: {self.solverFailures}!!!\n\nnew positions[0]: {self.positions[0]}\n\n')
                else:
                    raise RuntimeError(f'Solver Failed {self.solverFailures} times')

        output=[]

        sSensors = self.mbs.variables['sSensors']
        sReference = self.mbs.variables['sReference']

        if self.outputVectorPattern == 1:
            for i in range(self.nStepsTotal):
                timestepData = []
                for c in range(len(sSensors)):
                    dataSensor = self.mbs.GetSensorStoredData(sSensors[c])[i+1][1:4] #i+1 to ignore the first step, the prestep function is run the first time at the second step, the simulation has nTotalSteps + 1 steps
                    dataReference = self.mbs.GetSensorStoredData(sReference[c])[i+1][1:4]
                    timestepData.append(dataSensor-dataReference) #store the deformaiton
                output.append(timestepData)
        elif self.outputVectorPattern == 0:
            for i in range(self.nStepsTotal):
                timestepData = []
                singleSensorModeIndex = self.mbs.variables['singleSensorModeIndex'] 
                dataSensor = self.mbs.GetSensorStoredData(sSensors[singleSensorModeIndex])[i+1][1:4]
                dataReference = self.mbs.GetSensorStoredData(sReference[singleSensorModeIndex])[i+1][1:4]
                timestepData = (dataSensor-dataReference)[2] #store the deformaiton in the sensors z direction
                output.append(timestepData)

        
        output = np.asarray(output)

        if solutionViewer:
            self.SC.visualizationSettings.nodes.show = False
            self.SC.visualizationSettings.loads.drawSimplified = False

            self.SC.visualizationSettings.connectors.defaultSize = 0.01
            self.SC.visualizationSettings.connectors.show = False

            self.SC.visualizationSettings.markers.show = False

            self.SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
            self.SC.visualizationSettings.contour.outputVariableComponent = -1 #y-component
            self.SC.visualizationSettings.bodies.deformationScaleFactor = 1

            self.SC.visualizationSettings.sensors.show = True
            self.SC.visualizationSettings.sensors.drawSimplified = False
            self.SC.visualizationSettings.sensors.showNumbers = True

            self.mbs.SolutionViewer()

        return output
    
    def ConvertToNNInputOutput(self, simulationInput, simulationOutput, hiddenInput=None, scale=True):
        nStepsTotal = simulationInput.shape[0]
        nPairs = math.floor((nStepsTotal-self.NNInputOverlap) / (self.NNInputLength-self.NNInputOverlap))

        if scale:
            InScaleVec = self.GetInputScaling()
            OutScaleVec = self.GetOutputScaling()
        else:
            InScaleVec = np.ones(self.GetInputScaling().shape)
            OutScaleVec = np.ones(self.GetOutputScaling().shape)

        nStepsForward = self.NNInputLength-self.NNInputOverlap
        NNInputs = []
        NNOutputs = []
        NNHiddens = []
        for i in range(nPairs):
            startInput = int(i * nStepsForward)
            startOutput = int(startInput + (self.NNInputLength-self.NNOutputLength))
            NNInputs.append(simulationInput[startInput:startInput+self.NNInputLength] * InScaleVec)
            NNOutputs.append(simulationOutput[startOutput:startOutput+self.NNOutputLength] * OutScaleVec)
            NNHiddens.append([])

        NNInputs = np.asarray(NNInputs)
        NNOutputs = np.asarray(NNOutputs)
        NNHiddens  = np.asarray(NNHiddens)
        return NNInputs, NNOutputs, NNHiddens
    
    # Sets the Trajectory (self.positions and self.rotations) form an Simulation Input vector
    def SetTrajectory(self, vec):
        self.positions=[]
        self.rotations=[]

        dataDict = {}

        entryLength = 12
        data = np.array(vec)

        if data.shape[1] != entryLength:
            raise ValueError(f"Shape of input Vector is invalid! ({data.shape})")

        dataDict['positions'] = []
        dataDict['rotations'] = []
        for d in data:
            dataDict['positions'].append(np.array(d[0:3]))
            dataDict['rotations'].append((np.array(d[3:12])).reshape(3,3))
                    

        vecLength = len(dataDict['positions'])
        lengthFactor = vecLength / self.nStepsTotal
        h = self.endTime/self.nStepsTotal

        for i in range(self.nStepsTotal):
            relIndex = lengthFactor * i
            iUpper = math.ceil(relIndex)
            iLower = math.floor(relIndex)

            if iLower < 0:
                iLower = 0

            # if the value is larger than the largest value of the list, iUpper will be == len(self.trajectoryTiems); that is one larger than the index of the last element. iLower will be set correctly, i Upper needs to be decremented by 1:
            if iUpper >= vecLength:
                iUpper = vecLength - 1
            
            pCurrent = tf.LinearInterpolateArray(relIndex, 0, lengthFactor * (self.nStepsTotal-1), dataDict['positions'][iLower], dataDict['positions'][iUpper])
            RCurrent = tf.LinearInterpolateArray(relIndex, 0, lengthFactor * (self.nStepsTotal-1), dataDict['rotations'][iLower], dataDict['rotations'][iUpper])

            self.positions.append(pCurrent)
            self.rotations.append(RCurrent)
    
    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()

        p0 = np.array(self.positions[0]).flatten()
        R0 = np.array(self.rotations[0])
        
        createSatellite(self.mbs, self.fem, p0, R0)
         
        h=self.endTime / self.nStepsTotal
        

        def PreStepFunction(mbs, t):
            stepCount = int(round(t/h))-1
            mActor = mbs.variables['mFlangeConstraint']

            p = self.positions[stepCount]
            R = self.rotations[stepCount]
            mbs.SetMarkerParameter(mActor, 'localPosition', p)
            mbs.SetObjectParameter(mbs.variables['cFlange'], 'rotationMarker0', R)
            return True
    
        self.mbs.SetPreStepUserFunction(PreStepFunction)
        self.mbs.Assemble()
        self.simulationSettings = exu.SimulationSettings()

        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime = self.endTime
        self.simulationSettings.solutionSettings.solutionWritePeriod = h
        self.simulationSettings.timeIntegration.newton.useModifiedNewton = True

        self.simulationSettings.solutionSettings.sensorsWritePeriod = h
        self.simulationSettings.solutionSettings.coordinatesSolutionFileName = "solution/satelite.txt"
        self.simulationSettings.solutionSettings.writeSolutionToFile=True
        
        self.simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.5 