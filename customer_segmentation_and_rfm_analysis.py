# ### Prepare and Understand Data (Data Understanding)
# 1. Load the dataset from the CSV file.
# 2. Display information about the dataset, including its shape, data types, and summary statistics.
# 3. Create new variables for the TOTAL NUMBER OF PURCHASE and TOTAL SPENDING for each customer.
# 4. Convert date columns in the dataset to the date format.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime as dt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1500)

df = pd.read_csv("data.csv")


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def check_df(dataframe):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Num of Unique ####################")
    print(dataframe.nunique())
    print("#################### Head ####################")
    print(dataframe.head())
    print("#################### Tail ####################")
    print(dataframe.tail())
    print("#################### NAN ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0.01, 0.05, 0.95, 0.99]).T)


check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_but_car = [col for col in cat_but_car if col != "master_id"]

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

correlation_matrix(df, num_cols)

df["total_transaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


# ### Calculating RFM Metrics
# 1. Define the Recency, Frequency, and Monetary metrics.
# 2. Calculate the Recency, Frequency, and Monetary metrics for each customer.
# 3. Assign these metrics to variables named recency, frequency, and monetary.
# 4. Change the names of the metrics to Recency, Frequency, and Monetary.

"""
Recency: Refers to the time elapsed since the customer's last purchase date. The shorter this period is, the "fresher" 
the customer is, indicating they are more active.

Frequency: Refers to the number of purchases a customer makes within a specific time frame. This reflects how often a 
customer buys, and thus, how frequently the brand interacts with them.

Monetary: Represents the total spending amount a customer makes within a specific time frame. This indicates how 
valuable a customer is and reflects how much revenue the brand generates.
"""

last_order_date = df["last_order_date"].max()  # Find the last order date
analysing_date = dt.datetime(2021, 6, 2)  # Setting the recency date for 2 days after the last order date

rfm_df = df.groupby("master_id")\
    .agg({"last_order_date": lambda last_order_date: (analysing_date - last_order_date.max()).days,
          "total_transaction": lambda total_transaction: total_transaction.sum(),
          "total_value": lambda total_value: total_value.sum()})

rfm_df.columns = ["receny", "frequency", "monetary"]
rfm_df.reset_index()


# ### Calculating RF and RFM Scores
# 1. Convert the Recency, Frequency, and Monetary metrics into scores between 1 and 5 using quantiles (qcut).
# 2. Create new columns recency_score, frequency_score, and monetary_score to store the scores.
# 3. Calculate the RFM score by combining recency_score and frequency_score as a single variable named RFM_score.

rfm_df["recency_score"] = pd.qcut(rfm_df["receny"], 5, [5, 4, 3, 2, 1])
rfm_df["frequency_score"] = pd.qcut(rfm_df["frequency"].rank(method="first"), 5, [1, 2, 3, 4, 5])
rfm_df["monetary_score"] = pd.qcut(rfm_df["monetary"], 5, [1, 2, 3, 4, 5])
rfm_df["RFM_score"] = rfm_df["recency_score"].astype(str) + rfm_df["frequency_score"].astype(str)


# ### Defining RF Scores as Segments
# 1. Define segments based on RFM scores using a segmentation map.
# 2. Create a new column segment to store the segment labels.

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at risk",
    r"[1-2]5": "can't lose them",
    r"3[1-2]": "about to sleep",
    r"33": "need attention",
    r"[3-4][4-5]": "loyal customers",
    r"41": "promising",
    r"[4-5][2-3]": "potential loyallists",
    r"51": "new customers",
    r"5[4-5]": "champions"
}

rfm_df["segment"] = rfm_df["RFM_score"].replace(seg_map, regex=True)
rfm_df["segment"].value_counts().plot(kind="bar")
plt.show(block=True)

segments = rfm_df['segment'].value_counts().sort_values(ascending=False)
data = segments.values
keys = segments.keys().values
palette_color = sns.color_palette('bright')
plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')
plt.show(block=True)


# ### Analyzing and Targeting Customer Segments
# 1. Analyze the averages of Recency, Frequency, and Monetary metrics for each segment.
# 2. Identify specific customer profiles for targeted marketing:
# a. Target customers who are loyal and female shoppers for a new women's shoe brand promotion.
# b. Target customers who are about to sleep or new customers for up to 40% discount promotions on Men's and Children's
# products.
# 3. Save the customer IDs of the selected profiles to CSV files.

segment_df = rfm_df.groupby("segment").agg({"receny": ["mean", "count"],
                                            "frequency": ["mean", "count"],
                                            "monetary": ["mean", "count"]})


final_df = pd.merge(df, rfm_df, on="master_id")

target_customers_A = final_df[(final_df["segment"].isin(["loyal customers", "champions"])) &
                              (final_df["interested_in_categories_12"].str.contains("KADIN"))]


target_customers_B = final_df[(final_df["segment"].isin(["about to sleep", "new customers"])) &
                              ((final_df["interested_in_categories_12"].str.contains("ERKEK") |
                                (final_df["interested_in_categories_12"].str.contains("COCUK") |
                                 (final_df["interested_in_categories_12"].str.contains("AKTIFCOCUK")))))]

target_customers_A["master_id"].to_csv("target_customers_A_id.csv", index=False)
target_customers_B["master_id"].to_csv("target_customers_B_id.csv", index=False)