# Customer Segmentation and RFM Analysis

## Project Overview
This project involves performing customer segmentation and RFM (Recency, Frequency, Monetary) analysis on customer data from a retail company. The primary goal is to categorize customers into segments based on their buying behavior and identify potential target groups for marketing campaigns.

## Dataset
The dataset used for this project is sourced from a CSV file named "flo_data_20k.csv."

## Task

### Prepare and Understand Data (Data Understanding)
1. Load the dataset from the CSV file.
2. Display information about the dataset, including its shape, data types, and summary statistics.
3. Create new variables for the total number of purchases and total spending for each customer.
4. Convert date columns in the dataset to the date format.

### Calculating RFM Metrics
1. Define the Recency, Frequency, and Monetary metrics.
2. Calculate the Recency, Frequency, and Monetary metrics for each customer.
3. Assign these metrics to variables named `recency`, `frequency`, and `monetary`.
4. Change the names of the metrics to `Recency`, `Frequency`, and `Monetary`.

### Calculating RF and RFM Scores
1. Convert the `Recency`, `Frequency`, and `Monetary` metrics into scores between 1 and 5 using quantiles (qcut).
2. Create new columns `recency_score`, `frequency_score`, and `monetary_score` to store the scores.
3. Calculate the RFM score by combining `recency_score` and `frequency_score` as a single variable named `rfm_score`.

### Defining RF Scores as Segments
1. Define segments based on RFM scores using a segmentation map.
2. Create a new column `segment` to store the segment labels.

### Analyzing and Targeting Customer Segments
1. Analyze the averages of Recency, Frequency, and Monetary metrics for each segment.
2. Identify specific customer profiles for targeted marketing:
   a. Target customers who are loyal and female shoppers for a new women's shoe brand promotion.
   b. Target customers who are about to sleep or new customers for up to 40% discount promotions on Men's and Children's products.
3. Save the customer IDs of the selected profiles to CSV files.

## How to Use
1. Clone this repository to your local machine.
2. Ensure you have Python and required libraries (pandas, numpy, matplotlib, seaborn, datetime) installed.
3. Run the Jupyter Notebook or Python script to execute the code.
4. Review the generated CSV files for targeted customer IDs.

Enjoy performing customer segmentation and RFM analysis with this project!
