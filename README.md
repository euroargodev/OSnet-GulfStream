## OSnet-Gulf_Stream

This repository contains the tools to make a prediction using the models computed for the GUlf Stream region (Pauthenet et al (in prep.)).
- Prediction_gridded.py is the routine for predicting profiles from a gridded input dataset. The outcome is an xarray object of four dimensional temperature and salinity.

- Preprocessing_input.ipynb provides the tools for converting the different surface inputs (SST, SLA, MDT,...) into Xarray objects to be read by the NN.

- MLD_adjustment.ipynb and MLD_adjustment_gridded.ipynb are the tools for adjusting the MLD after the prediction of the NN. It uses the prediction of the MLD to reduce the number of density inversions and improve the MLD prediction of the profiles.

