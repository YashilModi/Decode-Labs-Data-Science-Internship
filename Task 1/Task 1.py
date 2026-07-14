#EDA & Feature Engineering DecodeLabs Data Science Project 1
import os
import glob
import pandas as pd

#1. Load the dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #Had problems importing the file
# Find the Excel file in this folder automatically
search_pattern = os.path.join(SCRIPT_DIR, "*.xlsx")
excel_files_found = glob.glob(search_pattern)
if len(excel_files_found) == 0:
    raise FileNotFoundError("No .xlsx file found in the script folder.")
input_file_path = excel_files_found[0]
df = pd.read_excel(input_file_path)
print("1. Data loaded")
print("File used:", os.path.basename(input_file_path))
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
# 2. Handle missing data
# Only CouponCode has missing values. Store into own coloumn
print("2. Handling missing data")
missing_counts = df.isnull().sum()
print("Missing values per column:")
print(missing_counts[missing_counts > 0])
df["CouponCode"] = df["CouponCode"].fillna("No Coupon")
print("CouponCode missing values replaced with 'No Coupon'")
print("Remaining missing values in the whole dataset:", df.isnull().sum().sum())
print("")

# 3. Check for outliers (IQR method)
print("3. Checking for outliers")
columns_to_check = ["Quantity", "UnitPrice", "ItemsInCart"]
for column_name in columns_to_check:

    first_quartile = df[column_name].quantile(0.25)
    third_quartile = df[column_name].quantile(0.75)
    iqr_value = third_quartile - first_quartile
    lower_limit = first_quartile - (1.5 * iqr_value)
    upper_limit = third_quartile + (1.5 * iqr_value)
    is_too_low = df[column_name] < lower_limit
    is_too_high = df[column_name] > upper_limit
    is_outlier = is_too_low | is_too_high
    number_of_outliers = is_outlier.sum()
    print("")
    print("Column:", column_name)
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Outliers found:", number_of_outliers)
    if number_of_outliers > 0:
        df[column_name] = df[column_name].clip(lower=lower_limit, upper=upper_limit)
        print("Outliers were capped at the limit.")
    else:
        print("No outliers found, nothing to change.")
# Recalculate TotalPrice, matches Quantity x UnitPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
print("")
print("TotalPrice recalculated to stay consistent with Quantity and UnitPrice.")
print("")

# 4. Engineer new features
print("4. Creating new features")
# Feature 1: Month the order was placed
df["OrderMonth"] = df["Date"].dt.month
# Feature 2: Day of the week the order was placed
df["OrderWeekday"] = df["Date"].dt.day_name()
# Feature 3: Whether the order was placed on a weekend
day_number = df["Date"].dt.dayofweek  # Monday=0 ... Sunday=6
df["IsWeekend"] = 0
df.loc[day_number == 5, "IsWeekend"] = 1
df.loc[day_number == 6, "IsWeekend"] = 1
# Feature 4: Whether a coupon was used on the order
df["HasCoupon"] = 0
df.loc[df["CouponCode"] != "No Coupon", "HasCoupon"] = 1
# Feature 5: Average value per item in the cart
df["AvgItemValue"] = df["TotalPrice"] / df["ItemsInCart"]
df["AvgItemValue"] = df["AvgItemValue"].round(2)
# Feature 6: Whether the order counts as "high value"
high_value_cutoff = df["TotalPrice"].quantile(0.75)
df["IsHighValueOrder"] = 0
df.loc[df["TotalPrice"] > high_value_cutoff, "IsHighValueOrder"] = 1
print("New features created:OrderMonth,OrderWeekday,IsWeekend,HasCoupon,AvgItemValue,IsHighValueOrder")
print("")
print("High value cutoff used:", round(high_value_cutoff, 2))
print("Orders marked as high value:", df["IsHighValueOrder"].sum())
print("Orders that used a coupon:", df["HasCoupon"].sum())
print("")

# 5. Save the cleaned dataset
output_file_path = os.path.join(SCRIPT_DIR, "cleaned_dataset.xlsx")
df.to_excel(output_file_path, index=False)