## Info
This code was written for my a bachelor thesis at the university of Innsbruck. It is based on the SLIDE method.

The following versions of packages were used: 
* Python 3.11.13  
* PyTorch 2.8.0 (+cu128)  
* exudyn 1.9.0  
* matplotlib 3.10.1  
* numpy 1.26.4  
* ngsolve 6.2.2501  
* spatialmath 1.1.14  

The `requirements.txt` file can be used to install the requirements with  
`pip install -r requirements.txt`   

### SLIDE
The method SLiding-window Initially-truncated Dynamic-response Estimator (SLIDE) is a deep-learning based method for estimating the output of mechanical and multibody systems.   
The corresponding research paper is now available on arXiv, doi: [arXiv.2409.18272](https://doi.org/10.48550/arXiv.2409.18272), and in the submission process for journal publication. 

### Exudyn
Exudyn is a flexible multibody dynamics simulation code. The C++ core ensures efficient simulation, while the Python interface enables compatibility with pytorch and other machine learning tools. 
For more information see the [extensive documentation](https://exudyn.readthedocs.io/en/latest/docs/RST/Exudyn.html) and the [examples](https://github.com/jgerstmayr/EXUDYN/tree/master/main/pythonDev/Examples) on github.

## Licence 
See [Licence.txt](Licence.txt).