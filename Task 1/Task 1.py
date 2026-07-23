import os

import pandas as pd

# always work from the folder this script is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#1 Load the orders data
orders = pd.read_excel("Dataset for Data Analytics.xlsx")
print("1. Data loaded")
print("Number of rows:", orders.shape[0])
print("Number of columns:", orders.shape[1])

#2 Fill in the missing coupon codes
print("2. Handling missing data")
missing_counts = orders.isna().sum()
print("Missing values per column:")
print(missing_counts[missing_counts > 0])
orders["CouponCode"] = orders["CouponCode"].fillna("No Coupon")
print("Remaining missing values in the whole dataset:", orders.isna().sum().sum())
print()

#3 Cap outliers with the IQR rule and redo TotalPrice
print("3. Checking for outliers")
for column in ["Quantity", "UnitPrice", "ItemsInCart"]:
    q1 = orders[column].quantile(0.25)
    q3 = orders[column].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outliers = (orders[column] < lower_limit) | (orders[column] > upper_limit)
    outlier_count = outliers.sum()

    # Second opinion: flag anything more than 3 standard deviations from the mean
    z_scores = (orders[column] - orders[column].mean()) / orders[column].std()
    z_outlier_count = (z_scores.abs() > 3).sum()

    print()
    print("Column:", column)
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Outliers found (IQR):", outlier_count)
    print("Outliers found (Z-score > 3):", z_outlier_count)

    if outlier_count > 0:
        orders[column] = orders[column].clip(lower_limit, upper_limit)

orders["TotalPrice"] = orders["Quantity"] * orders["UnitPrice"]
print()

#4 Add the date, coupon and order value columns
print("4. Creating new features")
orders["OrderMonth"] = orders["Date"].dt.month
orders["OrderWeekday"] = orders["Date"].dt.day_name()
orders["IsWeekend"] = (orders["Date"].dt.dayofweek >= 5).astype(int)
orders["HasCoupon"] = (orders["CouponCode"] != "No Coupon").astype(int)
orders["AvgItemValue"] = (orders["TotalPrice"] / orders["ItemsInCart"]).round(2)

high_value_cutoff = orders["TotalPrice"].quantile(0.75)
orders["IsHighValueOrder"] = (orders["TotalPrice"] > high_value_cutoff).astype(int)

print("High value cutoff used:", round(high_value_cutoff, 2))
print("Orders marked as high value:", orders["IsHighValueOrder"].sum())
print("Orders that used a coupon:", orders["HasCoupon"].sum())
print()

#5 Save the cleaned file
print("5. Saving the cleaned dataset")
orders.to_excel("cleaned_dataset.xlsx", index=False)
print("Saved to: cleaned_dataset.xlsx")
