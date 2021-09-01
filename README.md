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

[Instructions for creation of STQS model.ipynb](Instructions for creation of STQS model.ipynb)

Sample data for running the codes can be found [here](https://www.dropbox.com/sh/160g84gkqh345wk/AADE_Gyyfla6yyCGcyWmjS5sa?dl=0). This folder contains sample input data files for a few .edf signals and some fake saved models (trained for only one epoch). This is shown just for the purpose of giving an idea of how the sample inputs and saved models look.<br\>

**Using the model**

All original STQS trained models (on SHHS) can be found in 'trained_models' folder. These models can be used using the test.py code to test it on any signal in hdf5 format.

**Citation**
This is the original implementation of the work: [STQS: Interpretable multi-modal Spatial-Temporal-seQuential model for automatic Sleep scoring](https://www.sciencedirect.com/science/article/pii/S0933365721000312).
```
@article{pathak2021stqs,
  title={STQS: Interpretable multi-modal Spatial-Temporal-seQuential model for automatic Sleep scoring},
  author={Pathak, Shreyasi and Lu, Changqing and Nagaraj, Sunil Belur and van Putten, Michel and Seifert, Christin},
  journal={Artificial Intelligence In Medicine},
  volume={114},
  pages={102038},
  year={2021},
  publisher={Elsevier}
}
```
