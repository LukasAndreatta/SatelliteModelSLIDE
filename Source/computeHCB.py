#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           computeHCB
#
# Details:  Script used for modal reduction, the result is saved to a .npy
#           file
#
# Author:   Lukas Andreatta
# date:     2025-09-15
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import exudyn as exu
from exudyn.utilities import *
from exudyn.FEM import *
import time
import ngsolve as ngs
from netgen.meshing import *
from netgen.csg import *

from SatilliteExudynModel import __getConnectionPointNodes

source = 1 # 0 - netgen, 1 abqus
if source == 0:
    outFileName='modelData/panel-netgen.npy'
elif source == 1:
    baseName='modelData/PanelV3-2_5mm' #files should be called baseName + '.inp', baseName + '_STIF.mtx', and baseName + '_MASS.mtx'
    outFileName='modelData/panel-abaqus.npy'
else:
    raise ValueError('invalid source, source must be 0 (netgen) or 1 (abaqus)')
##Geometry:
#panel
w1 = 0.2    #m panel side length
w2 = w1     #m panel side length
tp = 0.01  #m panel thickness
#hinge
h1 = 0.01   #m hinge width
h2 = 0.025  #m hinge width
ho = 0.0075 #m hinge hinge offset
#Materials density kg/m3, EModulus Pa, poissonsRatio
aluminium = (2.699e3, 70200e6, 0.345)
acrylic = (1.19e3, 3300e6, 0.35)
##Mesh
h_max=0.005 #m max length for mesh in 
quad_dominated=False
meshOrder = 1
##HCB
nModes = 64
constraint = 3 # 1 = hinge 1 only, 2 = hinge 2 only 3 = both hinges

if source == 0:
    geo = CSGeometry()
    panelMain = OrthoBrick(Pnt(0, -w2/2, 0), Pnt(w1, w2/2, tp))
    hinge1 = OrthoBrick(Pnt(w1+h1, w2/2-ho-h2, tp),Pnt(w1, w2/2-ho, 0))
    hinge2 = OrthoBrick(Pnt(w1, -w2/2+ho, 0), Pnt(w1+h1, -w2/2+ho+h2, tp))
    geo.Add(panelMain)

    geo.Add(hinge1)
    geo.Add(hinge2)



    mesh = ngs.Mesh(geo.GenerateMesh(maxh=h_max, quad_dominated=quad_dominated))
    mesh.Curve(1)

    if False:  # visualise the mesh inside netgen/ngsolve
        import netgen.gui
        ngs.Draw(mesh)
        for i in range(10000000):
            netgen.Redraw()
            time.sleep(0.05)

    fem = FEMinterface()
    meshData_panelMain = fem.ImportMeshFromNGsolve(mesh,
                                                useElementSets=True,
                                                density=acrylic[0],
                                                youngsModulus=acrylic[1],
                                                poissonsRatio=acrylic[2],
                                                meshOrder=meshOrder)
if source == 1:
    fem = FEMinterface()
    nodes=fem.ImportFromAbaqusInputFile(baseName+'.inp', typeName='Instance', name='satellite')
    fem.ReadMassMatrixFromAbaqus(baseName+'_MASS.mtx')
    fem.ReadStiffnessMatrixFromAbaqus(baseName+'_STIF.mtx')

n1 = [1,0,0]

pLeft  = [0,0,0]
pRight = [w1+h1,0,0]

nodesL_panelMain = fem.GetNodesInPlane(pLeft,n1)
weightsL_panelMain = fem.GetNodeWeightsFromSurfaceAreas(nodesL_panelMain)

nodesR_panelMain = fem.GetNodesInPlane(pRight,n1)
weightsR_panelMain = fem.GetNodeWeightsFromSurfaceAreas(nodesR_panelMain)


##Get connectors
u = w2/2-ho
l = w2/2-ho-h2
nll, wll = __getConnectionPointNodes(fem, nodesL_panelMain, -l, -u, 1)
nlu, wlu = __getConnectionPointNodes(fem, nodesL_panelMain, u, l, 1)
nrl, wrl = __getConnectionPointNodes(fem, nodesR_panelMain, -l, -u, 1)
nru, wru = __getConnectionPointNodes(fem, nodesR_panelMain, u, l, 1)


##HCB
if constraint == 1:
    boundaryNodesList = [nrl]
elif constraint == 2:
    boundaryNodesList = [nru]
elif constraint == 3:
    boundaryNodesList = [nrl, nru]
else:
    raise ValueError('constraint must be 1, 2, or 3 no other values allowed')

print("panel nNodes=", fem.NumberOfNodes())
print("compute HCB modes... ")
start_time = time.time()
fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=[nrl,nru],
                                  nEigenModes=nModes,
                                  useSparseSolver=True,
                                  computationMode=HCBstaticModeSelection.RBE2)

fem.CheckConsistency()
exu.Print("HCB modes needed %.3f seconds" % (time.time() - start_time))
cms = ObjectFFRFreducedOrderInterface(fem)

fem.SaveToFile(outFileName)