#!/bin/bash

dataset_name='towhidultonmoy/cicids2018'
zip_name=$(basename "$dataset_name")

# Create the data directory
mkdir -p ./kaggle_data

# Download the dataset from Kaggle (uncomment the next line if you have the Kaggle API key configured)

kaggle datasets download -d "$dataset_name"

# Unzip the downloaded dataset
unzip "./$zip_name.zip" -d ./kaggle_data

# Remove the downloaded zip file (uncomment the next line if you want to remove it)
# rm -rv "./$zip_name.zip"

rm -rv ./kaggle_data/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
