# By Lukas Andreatta
import numpy as np

# linear interpolation between two numpy arrays
# adapted from exudyn SmoothStep  
def LinearInterpolateArray(x, x0, x1, value0, value1):
    loadValue = value0

    if x > x0:
        if x < x1:
            dx = x1-x0
            loadValue = ((x1-x)*value0 + (x-x0)*value1)/dx
        else:
            loadValue = value1
    return loadValue

# interpolation between two numpy arrays
# interpolation function should be some function f(x) -> R usually with:
#       f(0) = 0
#       f(1) = 1
# if these conditions are met the return for x <= x0 will be value0 and for x >= x1 value1
# if not then return outside the interval x0, x1 will be the same as the value of the closest value in the interval
def generalInterpolate(x, x0, x1, value0, value1, interpolationFunction):
    if x < x0:
        x = x0
    elif x > x1: 
        x = x1

    dx = x1-x0
    xp = interpolationFunction((x-x0)/dx)
    return ((1-xp)*value0 + (xp)*value1)

## functions to use for time scaling with generalInterpolate.
def linearFunction(x):
    return(x)

def sinSmooth(x):
    return(-0.5 * np.cos(x*np.pi) + 0.5)

def parabolicSmooth(x):
    if x < 0.5:
        return 2 * x**2
    else:
        return -2 * (x-1) ** 2 + 1

# generates the interpolation function for a trapezoidal trajectory, the function will start at 0 with a parabola the smoothly transition too a linear function at x = ratio/2, then transition back to a parabola at x = 3*ratio/2. The function starts at 1 and ends at 0. The first derivative is 0 at x = 0 and x = 1, and it is continuous.
# ratio give the ratio of the 2 parabola too linear part. It must be between 0 and 1.
def generateTrapezoidalSmoothFunction(ratio):
    if ratio < 0 or ratio > 1:
        raise ValueError(f'ratio (={ratio}) must be between 0 and 1')
    
    # would result in div by 0 error
    if ratio == 1:
        return linearFunction

    parLength = 0.5 * (1 - ratio)
    linSlope = 2/(1+ratio)
    parSlope = (linSlope)/(1-ratio)

    xTransition1 = parLength
    yTransition1 = parSlope * xTransition1 ** 2
    xTransition2 = parLength + ratio
    yTransition2 = yTransition1 + linSlope * (xTransition2 - xTransition1)

    def trapezoidalSmooth(x):
        if x < xTransition1:
            return parSlope * x**2
        elif x < xTransition2:
            return linSlope * (x-xTransition1) + yTransition1
        else:
            return -parSlope * (x-1) ** 2 + 1 

    return trapezoidalSmooth

# Used to generate an oscillating function, that starts at 0
# oscillations:     gives the number of oscillations in the interval 0,1 if the number is uneven then the return of the oscillation Function will be 1 at x = 1
def generateSinOscillationFunction(oscillations):
    def sinOscillationFunction(x):
        return(-0.5 * np.cos(x*np.pi*oscillations) + 0.5)
    return sinOscillationFunction

# Interpolates between several np Arrays one after the other.
# x:            float to determine where the function is evaluated
# pointXVals:   at which x a the corresponding point should be reached
# pointList:    a List of floats or np Arrays. This Function will interpolate between them
# interpolationFunction:    Function used to interpolate between the points.
def multiPointTrajectory(x, pointXVals, pointList, interpolationFunction):
    nPoints = len(pointList)

    if nPoints != len(pointXVals):
        raise ValueError('pointXVals and pointList must have the same length')
    
    if not np.all(pointXVals[:-1] <= pointXVals[1:]):
        raise ValueError('pointXVals, must listed in ascending order')

    try:
        pointList = np.array(pointList)
    except ValueError:
        raise ValueError('pointList is not valid')
    
    # find indices of the previous and the next point
    i0 = 0
    i1 = 1
    for i in range(nPoints -1):
        if x < pointXVals[i]:
            break
        else:
            i0 = i
            i1 = i + 1
    
    return generalInterpolate(x, pointXVals[i0], pointXVals[i1], pointList[i0], pointList[i1], interpolationFunction)