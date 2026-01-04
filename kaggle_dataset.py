import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("data/Mumbai House Prices.csv")
"""
print(df.head())

print("\n Missing values:\n", df.isna().sum())



print("\n Added price_per_sqft column:")
print(df.head())


avg_price = np.mean(df["price"])
median_price = np.median(df["price"])

print(f"\n Average Price: ₹{avg_price:.2f}")
print(f" Median Price: ₹{median_price:.2f}")
"""

def convert_price(row):
    price = float(row["price"])
    unit = str(row["price_unit"]).strip().lower()
    if pd.isna(price):
        return np.nan
    if unit in ["cr", "crore"]:
        return price * 1e7        
    elif unit in ["l", "lac", "lakh"]:
        return price * 1e5  
    else:
        return price   

df["price_inr"] = df.apply(convert_price, axis=1)     
    
df["price_per_sqft"] = df["price_inr"] / df["area"]

print(df[["region", "area", "price_inr", "price_per_sqft"]].head(10))

avg_rent_by_area = (df.groupby("region")["price_per_sqft"].mean().sort_values(ascending=False))

print("\n Average Rent per Sqft by Location:\n")
print(avg_rent_by_area)

df.to_csv("data/Mumbai House Prices.csv", index=False)