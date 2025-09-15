# Modified by Lukas Andreatta
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Library for creating multibody simulation models for the SLIDE method. 
#
# Author:   Peter Manzl, Johannes Gerstmayr
# Date:     2024-09-28
#
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.utilities import *
from exudyn.signalProcessing import GetInterpolatedSignalValue, FilterSignal
from exudyn.physics import StribeckFunction
from exudyn.utilities import SensorNode, SensorObject, Rigid2D, RigidBody2D, Node1D, Mass1D, GraphicsDataSphere, color4black
from exudyn.utilities import ObjectGround, VObjectGround, GraphicsDataOrthoCubePoint, color4grey, VCoordinateSpringDamper
from exudyn.utilities import NodePointGround, Cable2D, MarkerBodyPosition, MarkerBodyRigid, VMass1D, color4blue, MarkerNodeCoordinate
from exudyn.utilities import CoordinateSpringDamper, VObjectRigidBody2D, CoordinateConstraint, MarkerNodePosition, Point2D
from exudyn.utilities import RevoluteJoint2D, GraphicsDataRectangle, SensorUserFunction, Torque, Force, LoadCoordinate, copy
from exudyn.utilities import MassPoint2D, VCable2D, GenerateStraightLineANCFCable2D, color4dodgerblue, SensorBody, RotationMatrixZ
from exudyn.FEM import *

import random
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, StdDH2HT, VRobotBase, VRobotTool, VRobotLink
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP

import sys
import numpy as np
from math import sin, cos, pi, tan, exp, sqrt, atan2

from enum import Enum #for data types
import time

try: 
    import exudyn.graphics as graphics 
except: 
    print('exudyn graphics could not be loaded. Make sure a version >= 1.8.52 is installed.' )
    print('some simulation models may not work correctly. ')
    
try: 
    import ngsolve as ngs
    import netgen
    from netgen.meshing import *
    from netgen.geom2d import unit_square
    from netgen.csg import *
except: 
    print('warning: ngsolve/netgen could not be loaded, thus the flexible robot model does not work.')
    
class ModelComputationType(Enum):
    dynamicExplicit = 1         #time integration
    dynamicImplicit = 2         #time integration
    static = 3         #time integration
    eigenAnalysis = 4         #time integration

    #allows to check a = ModelComputationType.dynamicImplicit for a.IsDynamic()    
    def IsDynamic(self):
        return (self == ModelComputationType.dynamicExplicit or 
                self == ModelComputationType.dynamicImplicit)


def AccumulateAngle(phi): 
    phiCorrected = copy.copy(phi)
    for i in range(1, len(phiCorrected)): 
        if abs(phiCorrected[i] - phiCorrected[i-1]) > np.pi: 
            phiCorrected[i:] += 2*np.pi * np.sign(phiCorrected[i-1] - phiCorrected[i])
    return phiCorrected

# 
def CreateVelocityProfile(tStartup, tEnd, nStepsTotal, vMax, aMax, nPeriods = [20, 60], flagNoise=True, trajType = 0): 
    
    v = np.zeros(nStepsTotal)
    h = tEnd / nStepsTotal
    nStartup = int(np.ceil(tStartup/h))
    v_start = np.random.uniform(-vMax, vMax)
    v[0:nStartup] = np.linspace(0, v_start, nStartup)    
    i_0 = nStartup
    i_E = i_0

    # if activated 50% chance to add noise
    if trajType  == 0: 
        if np.random.random() > 0.5 and flagNoise: 
            addNoise = True
        else: 
            addNoise = False
            
        while i_E < nStepsTotal: 
            i_E += int(np.random.uniform(nPeriods[0], nPeriods[1]))
            # print('idiff: ', i_diff)
            if i_E > nStepsTotal: 
                i_E = nStepsTotal
            di = i_E - i_0
            dt = di * h
            a_ = np.random.uniform(-aMax, aMax)
            if np.random.random() < 0.1:  # 10% are constant
                a_ = 0
                
            dv = a_*dt
            if abs(v[i_0-1] + dv) > vMax: 
                dv = vMax * np.sign(v[i_0-1] + dv)  - v[i_0-1]
                if a_ == 0: 
                    continue
                
                dt = dv / a_                
                i_E = i_0 + int(dt / h)
                if dt < h*10: 
                    continue
                if i_E > nStepsTotal: 
                    i_E = nStepsTotal
            if dt > nPeriods[1]*h: 
                dt = nPeriods[1]*h
            if (i_E - i_0 + 1) < 0: 
                continue
            v[i_0-1:i_E] = v[i_0-1] + np.linspace(0, a_*dt, i_E - i_0 +1)
            maxNoise = 0
            if addNoise: 
                maxNoise = 0.04 * vMax
                nVal = len(v[i_0-1:i_E])
                v[i_0-1:i_E] += ((np.random.random(nVal) - 0.5)* maxNoise) * (np.random.random(nVal) > 0.5)
            if abs(v[i_E-1]) > (vMax + bool(flagNoise) * maxNoise + 1e-12): 
                if a_ == 0:
                    continue
                i_max = i_0 + (vMax - v[qi_0]) / a_
                print('Velocity too high at {}: {}'.format(i_E, v[i_E-1]))
            # print('traj length: {} steps'.format(i_E - i_0))
            i_0 = i_E
        v = np.clip(v, -vMax, vMax) # avoid values above vMax
        
    elif trajType  == 1: 
        v[i_0:i_0+nStartup] = v[i_0-1]
        i_0 = i_0+nStartup 
        n2 = int(np.ceil(np.random.uniform(nPeriods[0], nPeriods[1])))
        i_E = i_0 + n2
        v2 = np.random.uniform(-vMax, vMax)
        dt2 = n2 * tEnd/nStepsTotal
        if abs(v[i_0-1] - v2) > aMax * dt2:
            v2 = v[i_0-1] + np.sign(v2 - v[i_0-1]) * dt2 * aMax
        v[i_0:i_E] = np.linspace(v[i_0-1], v2, n2)
        v[i_E:] = v[i_E-1]
        if flagNoise: 
            maxNoise = 0.04 * vMax
            v += ((np.random.random(nStepsTotal) - 0.5)* maxNoise) * (np.random.random(nStepsTotal) > 0.5)
        
    else: 
        raise ValueError('CreateVelocityProfile does not support trajType {}'.format(trajType ))
    return v


# %timeit numericIntegrate(v, dt): 
# 33.9 µs ± 233 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
# According to timeit faster than zero allocation (38 µs ± 141 ns per loop) and
# using np.tri version (72.2 µs ± 1.79 µs per loop); probably because of big 
# memory allocation for larger v vectors. Note that for shorter vectors of v
# the version with the np.tri becomes faster: for len(v) == 100
# 17.1 µs ± 156 ns --> 11.6 µs ± 88 ns.
def numericIntegrate(v, dt): 
    p = [0]
    for i in range(1, len(v)): 
        p += [p[-1] + v[i] * dt]
    return p
    # return np.tri(len(v), len(v)) @ v * dt


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
# base class for creating an Exudyn model
class SimulationModel():

    #initialize class 
    def __init__(self):
        self.SC = None
        self.mbs = None
        self.modelName = 'None'
        self.modelNameShort = 'None'
        self.inputScaling = np.array([])
        self.outputScaling = np.array([])
        self.inputScalingFactor = 1. #additional scaling factor (hyper parameter)
        self.outputScalingFactor = 1.#additional scaling factor (hyper parameter)
        self.nOutputDataVectors = 1  #number of output vectors (e.g. x,y,z)
        self.nnType = 'unused' #  this is currently not used. The simulation model should be seperated from the neural network model. 
        
        self.nStepsTotal = None
        self.computationType = None
        #number of ODE2 states
        # ODE2 states are coordinates which are describes by differential equations of second order. 
        self.nODE2 = None               
        self.useVelocities = None
        self.useInitialValues = None
        
        self.simulationSettings = None
    
    # warning: depreciated!     
    def NNtype(self):
        return self.nnType
    def IsRNN(self):
        return self.nnType == 'RNN'
    def IsFFN(self):
        return self.nnType == 'FFN'
    def IsCNN(self): 
        return self.nnType == 'CNN'
    
    #create a model and interfaces
    def CreateModel(self):
        pass

    #get model name
    def GetModelName(self):
        return self.modelName

    #get short model name
    def GetModelNameShort(self):
        return self.modelNameShort

    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal

    #return a numpy array with additional scaling for inputs when applied to mbs (NN gets scaled data!)
    #also used to determine input dimensions
    def GetInputScaling(self):
        return self.inputScalingFactor*self.inputScaling
    
    #return a numpy array with scaling factors for output data
    #also used to determine output dimensions
    def GetOutputScaling(self):
        return self.outputScalingFactor*self.outputScaling
    
    # Set the input scaling
    def SetInputScaling(self, inputScalingVector, inputScalingFactor = 1):
        self.inputScaling = inputScalingVector
        self.inputScalingFactor = inputScalingFactor
    
    # Set the output scaling
    def SetOutputScaling(self, outputScalingVector, outputScalingFactor = 1):
        self.outputScaling = outputScalingVector
        self.outputScalingFactor = outputScalingFactor

    #return input/output dimensions [size of input, shape of output]
    def GetInputOutputSizeNN(self):
        return [self.inputScaling.shape, self.outputScaling.shape]
    
    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return np.array([])

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        return np.array([])

    #create initialization of (couple of first) hidden states
    def CreateHiddenInit(self, isTest):
        return np.array([])
    
    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        return {'data':None}
    
    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        return {'ODE2':[]}

    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        return np.array([])

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        return {}

    #get compute model with given input data and return output data
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False, cnt=None):
        return np.array([])
    
    #by Lukas Andreatta
    #Takes the input for the Exudyn simulation, and its output and converts it to test or training samples. That is input for a Neural Network, with the output truncated and shifted. Input and output are scaled. The return is one \mipy{np.array} with The Neural Network input vectors and one with the Neural Network output vectors. One Exudyn Simulation can (and usually is) used to create multiple vector pairs for Neural Network training.
    def ConvertToNNInputOutput(self, simulationInput, simulationOutput, hiddenInput):
        NNInputs = np.array([])
        NNOutputs = np.array([])
        NNHiddens = np.array([])
        return NNInputs, NNOutputs, NNHiddens 
    
    #visualize results based on given outputData
    #outputDataColumns is a list of mappings of outputData into appropriate column(s), not counting time as a column
    #  ==> column 0 is first data column
    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        nColumns = self.nODE2
        data = self.OutputData2PlotData(outputData, forSolutionViewer=True)
        
        # columnsExported = dict({'nODE2':self.nODE2, 
        #                         'nVel2':0, 'nAcc2':0, 'nODE1':0, 'nVel1':0, 'nAlgebraic':0, 'nData':0})
        columnsExported = [nColumns, 0, 0, 0, 0, 0, 0] #nODE2 without time
        if data.shape[1]-1 != nColumns:
            raise ValueError('SimulationModel.SolutionViewer: problem with shape of data: '+
                             str(nColumns)+','+str(data.shape))

        nRows = data.shape[0]
        
        
        sol = dict({'data': data, 'columnsExported': columnsExported,'nColumns': nColumns,'nRows': nRows})

        self.mbs.SolutionViewer(sol,runOnStart=True)