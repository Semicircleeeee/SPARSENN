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

#### **data-preprocessing**


```pythonscript
data_preprocessing(pos=pos, neg=neg, 
                       pos_adductlist=["M+H","M+NH4","M+Na","M+ACN+H","M+ACN+Na","M+2ACN+H","2M+H","2M+Na","2M+ACN+H"], 
                       neg_adductlist = ["M-H","M-2H","M-2H+Na","M-2H+K","M-2H+NH4","M-H2O-H","M-H+Cl","M+Cl","M+2Cl"], 
                       idx_feature = 4, match_tol_ppm=5, zero_threshold=0.75, log_transform=True, scale=1000)
```
- `pos`
(DataFrame) The dataframe of positive features. The first two columns should be m/z and time.

Default: None

- `neg`
(DataFrame) The dataframe of negative features. The first two columns should be m/z and time.

Default: None

- `pos_adductlist`
(list) The list of ion adduct for positive features.

Default: `["M+H","M+NH4","M+Na","M+ACN+H","M+ACN+Na","M+2ACN+H","2M+H","2M+Na","2M+ACN+H"]`

- `neg_adductlist`
(list) The list of ion adduct for negative features.

Default: `["M-H", "M-2H", "M-2H+Na", "M-2H+K", "M-2H+NH4", "M-H2O-H", "M-H+Cl", "M+Cl", "M+2Cl"]`

- `idx_feature`
(int) The number of columns of information for `pos` and `neg`. If the data only contains columns of `m/z` and `time`, the value should be 2.
Default: 4

- `match_tol_ppm`
(int) The tolerance of the maximum difference of m/z value between observation and known value in reference KEGG database during annotation equals `match_tol_ppm/1e6`. 
Default: 5

- `zero_threshold`
(float) The threshold of zero expression when selection features. Only keep features with no more than `zero_threshold` of zero expression. The value should be in range of (0, 1).
Default: 0.75

- `log_transform`
(boolean) Whether conduct (log+1) transformation to the feature data.
Default: True

- `scale`
(int) Whether to scale the feature data to be in range from [-scale/2, scale/2]. If `scale=False`, then no scaling will be conducted.
Default: 1000

#### **sparse_nn** 
```pythonscript
sparse_nn(expression, target, partition, feature_meta, sparsify_coefficient=0.3, threshold_layer_size=100, 
              num_hidden_layer_neuron_list=[20], drop_out=0.3, random_seed=10, 
              batch_size=32, lr=0.001, weight_decay=0, num_epoch=100)
```

# Tutorial 

For the step-by-step tutoral, please refer to the notebook:

https://github.com/tianlq-prog/SPARSENN/blob/master/Tutorial/example.ipynb

# Citation
