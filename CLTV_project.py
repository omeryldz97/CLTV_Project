#Task 1: Preparing the Data

#Step1: The process import
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
import pandas as pd
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format", lambda x:"%.5f" %x)

df_=pd.read_csv("") #Since the data is private, it is not shared.
df=df_.copy()
df.head()
df.isnull().sum()
#Step 2: Define the outlier threshold and replace_with_thresholds functions needed to suppress outliers.

def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
     dataframe.loc[(dataframe[variable] < low_limit), variable]=low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable]=up_limit

#Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online" If the variables have outliers, suppress them.
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

#Step 4: Omnichannel means that customers shop from both online and offline platforms. Create new variables for each customer's total purchases and spending.
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["order_num_total"]=df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df.head()

#Step 5: Examine the variable types. Change the type of variables that express date to date.
df.info()
my_date=["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[my_date]=df[my_date].apply(pd.to_datetime)

#Task 2: Creating the CLTV Data Structure
#Step 1: Take 2 days after the date of the last purchase in the data set as the analysis date.
df["last_order_date"].max() #2021-05-30
today_date=dt.datetime(2021,6,1)
#Step 2: Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values. Monetary value will be expressed as the average value per purchase, and recency and tenure values will be expressed in weekly terms.
new_df=pd.DataFrame({"Customer_ID":df["master_id"],"Recency_Cltv_weekly":((df["last_order_date"] - df["first_order_date"]).dt.days)/7,"T_Weekly":((today_date - df["first_order_date"]).astype("timedelta64[D]"))/7,"frequency":df["order_num_total"],"monetary_CLTV_avg":df["customer_value_total"]/df["order_num_total"]})
new_df.head()

#Görev 3: BG/NBD, Gamma-Gamma Establishment of Models and Calculation of CLTV
#Adım 1: BG/NBD fit the model.
bgf=BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(new_df["frequency"],new_df["Recency_Cltv_weekly"],new_df["T_Weekly"])
#Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.
new_df["exp_sales_3_month"]=bgf.predict(4*3,new_df["frequency"],new_df["Recency_Cltv_weekly"],new_df["T_Weekly"])

# Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.
new_df["exp_sales_6_month"]=bgf.predict(4*6,new_df["frequency"],new_df["Recency_Cltv_weekly"],new_df["T_Weekly"])
#Adım 2: Fit the Gamma-Gamma model. Estimate the value that the customers will leave on average and add the exp average value to the cltv dataframe nine.
ggf=GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(new_df["frequency"],new_df["monetary_CLTV_avg"])

#Step 3: Calculate the 6-month CLTV and add the dataframe with the name cltv.
cltv=ggf.customer_lifetime_value(bgf,new_df["frequency"],new_df["Recency_Cltv_weekly"],new_df["T_Weekly"],new_df["monetary_CLTV_avg"],time=6,freq="W",discount_rate=0.01)
new_df["CLTV"]=cltv
# Observe the 20 people with the highest Cltv value.
new_df.sort_values(by="CLTV",ascending=False).head(20)

#Task 4: Creating Segments by CLTV Value
#Step 1: Divide all your customers into 4 groups (segments) according to 6-month CLTV and add the group names to the dataset.
new_df["segment"]=pd.qcut(new_df["CLTV"],4,labels=["D","C","B","A"])
new_df.head()
