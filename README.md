# An integrated deep learning framework for the interpretation of untargeted metabolomics data
We introduce an integrated deep learning framework for metabolomics data that takes matching uncertainty into consideration. The model is devised with a gradual sparsification neural network based on the known metabolic network and the annotation relationship between features and metabolites. This well-designed architecture characterizes metabolomics data and reflects the modular structure of biological system. Three goals can be achieved simultaneously without requiring much complex inference and additional assumptions: (1) evaluate metabolite importance, (2) infer feature-metabolite matching likelihood, and (3) select disease sub-networks from the overall metabolic network. 

![workflow](https://github.com/tianlq-prog/SPARSENN/blob/master/docs/images/workflow.png)

# Dependencies
- numpy>=1.21.5
- pandas>=1.5.3
- python_igraph>=0.10.4
- scikit_learn>=1.0.2
- torch>=1.11.0

# Installation

## PypI

You can directly install the package from PyPI.

`pip install MetaMatching==0.2.3`

# Tutorial 

For the step-by-step tutoral, please refer to the notebook:

# Citation
