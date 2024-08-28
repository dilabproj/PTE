# PTE: Periodic Transformer Encoder for Short-term and Long-term Travel Time Prediction

This repository contains the code implementation for the PTE, a framework designed for short-term and long-term travel time prediction. The research leverages the Taiwan Expressway dataset provided by the Freeway Bureau of Taiwan, R.O.C. [1], and utilizes a transformer-based model to predict travel times.

## Dataset

- **Source:** Taiwan Expressway dataset from the [Freeway Bureau of Taiwan, R.O.C.](https://tisvcloud.freeway.gov.tw) [1].
- **Coverage:** October 1, 2019, to January 31, 2021.
- **Usage:** 
  - Training Data: First year of data.
  - Validation Data: Subsequent month.
  - Testing Data: Final three months.

## Data Preprocessing

The raw traffic data is processed through a series of steps to generate the `.npy` files required for model training. The preprocessing steps are based on methodologies detailed in [2]. Follow the steps below to prepare the data:

1. **Download Traffic Data:**
   - Script: `download_traffic_data.py`
   - This script downloads the raw traffic data from the source.

2. **Convert XML to CSV:**
   - Script: `traffic_xml_to_csv.py`
   - Converts the downloaded XML files into CSV format for easier processing.

3. **Generate Traffic Data:**
   - Script: `generate_traffic_data.py`
   - Processes the raw CSV files to generate cleaned and structured traffic data.

4. **Convert Raw CSV to DataFrame:**
   - Script: `csvraw2dataframe.py`
   - Converts the processed CSV files into Pandas DataFrames for further manipulation.

5. **Convert CSV to Numpy Array:**
   - Script: `csv2npy.py`
   - Converts the DataFrames into `.npy` files, which are the final input format for the PTE model.
   - The processed data will be saved in the `./nfb_final` directory, with filenames corresponding to the specific road segments.

## Model Training and Testing

- **Training and Testing Procedures:**
  - Notebook: `PTE.ipynb`
  - This notebook contains the full implementation for training and testing the PTE model on the processed data.

- **Model Definition:**
  - Script: `model.py`
  - Defines the architecture of the PTE model used in this study.

## Model Parameters

The trained model parameters are available due to their large size and are stored in a cloud directory. You can access them using the following link:

- Trained Model Parameters: [link](https://dilab.myds.me:49150/?launchApp=SYNO.SDS.App.FileStation3.Instance&launchParam=openfile%3D%252FDILab%25E5%2585%25B1%25E5%2590%258C%25E9%259B%25B2%25E7%25AB%25AF%252Fbest_models%252F&SynoToken=rDmppNGJdrDag)

After downloading the trained model parameters, place them in the `/best_models` directory to use them with the provided code.

## References

[1] Taiwan Expressway dataset, Freeway Bureau, Taiwan, R.O.C.  
[2] Y. Huang, H. Dai, and V. S. Tseng, “Periodic Attention-based Stacked Sequence to Sequence framework for long-term travel time prediction,” Knowledge-Based Systems, vol. 258, p. 109976, 2022.
