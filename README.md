# Credit Card Fraud Detection System

## Project Description
This project implements a hybrid approach to detect credit card fraud by combining unsupervised learning (Self-Organizing Maps - SOM) with supervised deep learning (Artificial Neural Network - ANN). The goal is to identify potential fraudulent patterns and build a predictive model based on the insights gained from the SOM.

## Problem Statement
Identify potential fraudulent credit card applications using a two-stage machine learning process.

## Methodology
The project is divided into three main parts:

### Part 1: Self-Organizing Maps (SOM)
*   **Goal:** Utilize SOM to map high-dimensional data onto a low-dimensional grid, which facilitates the visual identification of clusters and outliers (potential fraudulent transactions).
*   **Methodology:** A SOM is trained on the credit card application data. The 'winner' nodes in the SOM represent distinct clusters. By visualizing the Mean Interneuron Distance (MID) map, areas with high MID indicate boundaries between clusters or isolated data points, which are strong indicators of outliers. Customers mapped to these outlier nodes are identified as potential frauds.

### Part 2: Going from Unsupervised to Supervised Deep Learning
*   **Goal:** Transform the unsupervised anomaly detection problem into a supervised classification problem.
*   **Methodology:** Based on the output from the SOM, a new target variable is created. Customers identified as 'frauds' by the SOM are labeled as '1' (fraudulent), and all other customers are labeled as '0' (non-fraudulent). This newly labeled dataset becomes suitable for supervised machine learning.

### Part 3: Artificial Neural Network (ANN)
*   **Goal:** Train a supervised ANN model to predict credit card fraud using the labeled dataset generated in Part 2.
*   **Methodology:** A simple Artificial Neural Network (ANN) classifier is built and trained. The ANN learns the complex patterns from the SOM-labeled data to predict the likelihood of fraud for new credit card applicants.

## Setup and Dependencies
To run this notebook, you will need the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `minisom`
- `scikit-learn` (for `MinMaxScaler` and `StandardScaler`)
- `tensorflow` (for `tf.keras`)

You can install `minisom` using pip:
```bash
!pip install minisom
```

## Data
The project uses a CSV file named `Credit_Card_Applications.csv` which should be available in the same directory as the notebook. The dataset contains customer information and a target variable indicating whether the customer was approved.

## Usage
Execute the cells sequentially in the notebook. 
1.  **Part 1** will train the SOM and identify potential fraudulent customers.
2.  **Part 2** will prepare the data for supervised learning by labeling the identified frauds.
3.  **Part 3** will build and train an ANN to predict fraud, ultimately providing a list of customers with their associated fraud probabilities.

## Output
The notebook will output:
- A visualization of the SOM with highlighted fraudulent patterns.
- A list of `Fraud Customer IDs` identified by the SOM.
- A pandas DataFrame (`df_fraud`) showing customers with a high probability of fraud as predicted by the ANN, sorted by their fraud probability.
