## SLIDE

The method SLiding-window Initially-truncated Dynamic-response Estimator (SLIDE) is a deep-learning based method for estimating the output of mechanical and multibody systems.   
The corresponding research paper is now available on arXiv, doi: [arXiv.2409.18272](https://doi.org/10.48550/arXiv.2409.18272), and in the submission process for journal publication. 

In the development following versions of packages were used: 
* Python 3.11.8
* pytorch 2.2.1 (+cu121)
* exudyn 1.8.52 (and newer)
* matplotlib 
* numpy 1.23.5  
* ngsolve 6.2.2403

The according `requirements.txt` file can be used to install the requirements with  
`pip install -r requirements.txt`   
For the flexible 6R robot example additionally ngsolve is required. We used Version 6.2.2403. 

### Exudyn
Exudyn is a flexible multibody dynamics simulation code. The C++ core ensures efficient simulation, while the Python interface enables compatibility with pytorch and other machine learning tools. 
For more information see the [extensive documentation](https://exudyn.readthedocs.io/en/latest/docs/RST/Exudyn.html) and the [examples](https://github.com/jgerstmayr/EXUDYN/tree/master/main/pythonDev/Examples) on github.

## Licence 
See [Licence.txt](Licence.txt).