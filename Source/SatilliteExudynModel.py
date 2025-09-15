#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           SatilliteExudynModel
#
# Details:  File defining the function use to create the Exudyn Simulation
#           model in the SatelliteModel (SatelliteSlide.py) class
#
# Author:   Lukas Andreatta
# date:     2025-09-15
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import exudyn as exu
from exudyn.FEM import *

from exudyn.utilities import * #includes itemInterface and rigidBodyUtilities
import exudyn.graphics as graphics 

import numpy as np

# Function to get the mesh nodes in a plane, between two planes normal to the direction specified by directionIndex (0 for x, 1 for y, 2 for z)
def __getConnectionPointNodes(fem, nodes, upper, lower, directionIndex, tolerance=0.0005):

    positions = [fem.GetNodePositionsAsArray()[i] for i in nodes]
    outNodes = []
    for i in range(len(nodes)):
        pos = positions[i][directionIndex]
        if pos <= upper + tolerance and  pos >= lower - tolerance:
            outNodes.append(nodes[i])

    weights = fem.GetNodeWeightsFromSurfaceAreas(outNodes)
    return outNodes, weights

# function creating the satellite at the point p0, with rotation R0 (rotation matrix),
# in the system mbs, using the part fem as solar panel.
def createSatellite(mbs, fem, p0, R0):
    ##Geometry:
    #physics
    gVec = [0,0,-g]
    # gVec = [0,0,0]
    stiffnessProportionalDamping=0.01
    #panel
    w1 = 0.2    #m panel side length
    w2 = w1     #m panel side length
    tp = 0.005  #m panel thickness
    #center box
    c1 = w1
    c2 = c1
    c3 = c1
    #hinge
    h1 = 0.01   #m hinge width
    h2 = 0.025  #m hinge width
    ho = 0.0075 #m hinge hinge offset
    #Materials density kg/m3, EModulus Pa, poissonsRatio
    aluminium = (2.699e3, 70200e6, 0.345)
    acrylic = (1.19e3, 3300e6, 0.35)

    ##Load FEM
    cms = ObjectFFRFreducedOrderInterface(fem)

    n1 = [1,0,0]

    pLeft  = [0,0,0]
    pRight = [w1+h1,0,0]

    nodesH = fem.GetNodesInPlane(pRight,n1)
    nodesC = fem.GetNodesInPlane(pLeft,n1)

    u = w2/2-ho
    l = w2/2-ho-h2
    nh1, wh1 = __getConnectionPointNodes(fem, nodesH, -l, -u, 1)
    nh2, wh2 = __getConnectionPointNodes(fem, nodesH, u, l, 1)
    nc1, wc1 = __getConnectionPointNodes(fem, nodesC, -l, -u, 1)
    nc2, wc2 = __getConnectionPointNodes(fem, nodesC, u, l, 1)

    # correction if the nodes found, do not match the desired position of the rigid constraint
    c1Correction = np.array([0,-w2/2+ho+h2/2,tp/2])-fem.GetNodePositionsMean(nc1)
    c2Correction = np.array([0,w2/2-ho-h2/2,tp/2])-fem.GetNodePositionsMean(nc2)
    
    ##Crete Simulation
    oGround = mbs.AddObject(ObjectGround(referencePosition= [0,0,0]
                                        ))

    mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0,0,0]))

    # base
    T0 = HomogeneousTransformation(R0, p0)

    # center box 
    centerCubeDim = [c1, c2, c3]
    TMid = HTtranslate([0,0,c3/2]) #center of mass, body0

    iCube0 = InertiaCuboid(density=acrylic[0], sideLengths=centerCubeDim)
    graphicsBody0 = graphics.Brick(centerPoint=[0,0,0],size=centerCubeDim,
                                color=[.3,.3,.3,1])

    bCenterCube=mbs.CreateRigidBody(inertia = iCube0,
                        referencePosition = HT2translation(T0 @ TMid),
                        referenceRotationMatrix=HT2rotationMatrix(T0),
                        gravity = gVec,
                        graphicsDataList = [graphicsBody0])

    mFlange = mbs.AddMarker(MarkerBodyRigid(bodyNumber=bCenterCube, localPosition=[0,0,-c3/2]))
    ml1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=bCenterCube, localPosition=[-w1/2,-w2/2+ho+h2/2,-c3*0.5+tp*0.5]))
    ml2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=bCenterCube, localPosition=[-w1/2,w2/2-ho-h2/2,-c3*0.5+tp*0.5]))
    mr1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=bCenterCube, localPosition=[w1/2,-w2/2+ho+h2/2,-c3*0.5+tp*0.5]))
    mr2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=bCenterCube, localPosition=[w1/2,w2/2-ho-h2/2,-c3*0.5+tp*0.5]))

    # first panel
    #nodes for sensors
    posSensor = [[0,-w2/2,tp], [0,+w2/2,tp]]
    mbs.variables['sSensors'] = []
    mbs.variables['sReference'] = []
    def addPanels(nPanels, direction, connectorMarkerList):
        connectorMarkerList = (connectorMarkerList[0], connectorMarkerList[1])
        objListPanels = []
        for i in range(nPanels):
            basePos = [direction*((0.5*w1)+(1+i)*(w1+h1)), 0, 0]
            if direction == -1:
                baseRot = RotXYZ2RotationMatrix([0,0,0])
            elif direction == 1:
                baseRot = RotXYZ2RotationMatrix([0,0,np.pi])
            else:
                raise ValueError('direction must be 1 or -1')
        
            TBase = HomogeneousTransformation(baseRot, basePos)
            obj = cms.AddObjectFFRFreducedOrder(mbs, positionRef=HT2translation(T0 @ TBase),
                                        rotationMatrixRef=HT2rotationMatrix(T0 @ TBase),
                                        initialVelocity=[0, 0, 0],
                                        initialAngularVelocity=[0, 0, 0],
                                        gravity=gVec,
                                        stiffnessProportionalDamping=stiffnessProportionalDamping,
                                        color = [1,1,1,1])
            
            objListPanels.append(obj)
            mh1 = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=obj['oFFRFreducedOrder'], 
                                                meshNodeNumbers=np.array(nh1),
                                                weightingFactors=wh1, 
                                                ))
            mh2 = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=obj['oFFRFreducedOrder'], 
                                                meshNodeNumbers=np.array(nh2),
                                                weightingFactors=wh2, 
                                                ))
            mbs.AddObject(GenericJoint(markerNumbers=[connectorMarkerList[0], mh1]))
            mbs.AddObject(GenericJoint(markerNumbers=[connectorMarkerList[1], mh2]))

            mc1 = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=obj['oFFRFreducedOrder'], 
                                                meshNodeNumbers=np.array(nc1),
                                                weightingFactors=wc1,
                                                offset=c1Correction
                                                ))
            mc2 = mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=obj['oFFRFreducedOrder'], 
                                                meshNodeNumbers=np.array(nc2),
                                                weightingFactors=wc2, 
                                                offset=c2Correction
                                                ))
            connectorMarkerList = (mc1, mc2)

            # Add Sensor
            for pos in posSensor:
                node = fem.GetNodeAtPoint(pos)
                referencSensorPos = np.array(pos) + np.array([direction * (c1*1.5+h1 + i * (w1+h1)),0 ,-c3/2])
                referencSensorPos[1] = -1.0 * direction * referencSensorPos[1] # to account for the panels being rotated 180° on the right (positive x) side
                sSensor = mbs.AddSensor(SensorSuperElement(bodyNumber=obj['oFFRFreducedOrder'], 
                                        meshNodeNumber=node, #meshnode number!
                                        outputVariableType = exu.OutputVariableType.Position, 
                                        storeInternal=True ))
                sReference = mbs.AddSensor(SensorBody(bodyNumber=bCenterCube,
                                     localPosition=referencSensorPos,
                                     outputVariableType = exu.OutputVariableType.Position,
                                     storeInternal=True))
                mbs.variables['sSensors'] += [sSensor]
                mbs.variables['sReference'] += [sReference]   
            
            

        return(connectorMarkerList)

    addPanels(3,-1,(ml1,ml2))
    addPanels(3,1,(mr2,mr1))
    mbs.variables['singleSensorModeIndex'] = 4 # index of the sensor for single sensor mode in the sSensors and sReference list
    mFlangeConstraint = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=p0))
    mbs.variables['mFlangeConstraint'] = mFlangeConstraint
    mbs.variables['cFlange'] = mbs.AddObject(GenericJoint(markerNumbers=[mFlangeConstraint, mFlange],  
                                rotationMarker0=R0,
                                ))