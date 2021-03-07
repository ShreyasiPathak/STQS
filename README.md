# STQS

**STQS: Interpretable multi-modal Spatial-Temporal-seQuential model for automatic Sleep scoring**

STQS, abbreviation for Spatio-Temporal-seQuential-Sleep-scoring, is a deep learning based sleep scoring model for multi-modal, multi-channel input data, designed to account for spatial, temporal and sequential information in the signals. The model also handle the class imbalance in the data. Further, to provide an insight into the black-box, we investigate the model’s decision making process through post-hoc interpretability techniques:  frequency-domain occlusion, time-domain occlusion, and pattern visualization of temporal filters in the CNN.  The identified decision-relevant time-domain and frequency-domain features  aligned  well  with  expert  constructed  AASM  sleep  annotation  guide-lines.  Further, we extract the modality importance associated with the model’s decision to analyse the need of multi-modal models over single-channel ones. 

**Installation and Prerequisites**

The packages required for running STQS:
1. python 3.7.5
2. h5py 2.9.0
3. pytorch 1.3.0
4. pickle
5. sklearn 
6. numpy 1.17.3
7. matplotlib 3.1.3
8. cudatoolkit=9.2

The exact environment can be created using the following file:<br/>
```conda env create -f environment.yml```

**Obtaining Data**

SHHS data can be downloaded [here](https://sleepdata.org/datasets/shhs)

**Running**

The steps of running the codes for STQS model is explained in:<br/>
```Instructions for creation of STQS model.ipynb```
