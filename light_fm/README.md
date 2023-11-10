# LIGHTFM
2 methods have been used here. Firstly, implementing the code presented in the official github code of lightfm(https://github.com/lyst/lightfm/tree/master). Second the implementation of lightfm from scratch.

## Official code
>Check and understand the various losses being used, mostly logistic will not be used as the task is not that of binary prediction.

>Mainly check the losses of WARP and BPR.

>Change the code for fetching the dataset. Use custom function for preprocesisng the datset and on different MovieLens versions.
Install lightfm.

> Understand how 'item_features', 'item_feature_labels', 'item_labels' are created.

> Understand the code on Building datasets for creating the required dataset out of movielens.

```
pip install lightfm
```
Run the official code
```
python light_fm_official.py
```

## Our Implementation from scratch
>Simply copy paste the basic implementation of Building the dataset from the Official code.
>Understand the concept of +ve and -ve usage in lightfm.
Run the code
```
python light_fm.py
```