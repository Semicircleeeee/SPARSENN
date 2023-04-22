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

You can directly install the package from PyPI.

`pip install MetaMatching==0.2.3`

# Function in MetaMatching

|  **Function**  | **Description**                                                                                       |
| :--------------: | ----------------------------------------------------------------------------------------------------------- |
|  **data-preprocessing**  | Preprocssing of the raw feature data. Match the features to potential metabolites, obtain their uncertainty matching matrix and the metabolic network. The annotation is based on the m/z of features. |
| **sparse_nn** | Train the model and output the analysis result in folder 'res'.         |

- **data-preprocessing**


# Tutorial 

For the step-by-step tutoral, please refer to the notebook:

https://github.com/tianlq-prog/SPARSENN/blob/master/Tutorial/example.ipynb

# Citation
