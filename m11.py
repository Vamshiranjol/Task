import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv('amazon.csv')
df.head()
df.tail()
print(df)



# Convert rating to numeric (handling any non-numeric values)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Filter out products with no ratings
df_filtered = df.dropna(subset=["rating"])

# Get top 5 highest-rated products   
top_rated = df_filtered.nlargest(5, "rating")

# Get top 5 lowest-rated products
lowest_rated = df_filtered.nsmallest(5, "rating")

# Plot the ratings
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Highest rated products
ax[0].barh(top_rated["product_name"], top_rated["rating"], color="green")
ax[0].set_title("Top 5 Highest Rated Products")
ax[0].set_xlabel("Rating")
ax[0].invert_yaxis()  # Invert to show highest on top

# Lowest rated products
ax[1].barh(lowest_rated["product_name"], lowest_rated["rating"], color="red")
ax[1].set_title("Top 5 Lowest Rated Products")
ax[1].set_xlabel("Rating")
ax[1].invert_yaxis()

plt.tight_layout()
plt.show()


# Convert discount percentage to numeric
df["discount_percentage"] = df["discount_percentage"].str.replace("%", "").astype(float)

# Filter out missing values
df_discounted = df.dropna(subset=["discount_percentage"])

# Get top 5 highest and lowest discounted products
top_discounted = df_discounted.nlargest(5, "discount_percentage")
lowest_discounted = df_discounted.nsmallest(5, "discount_percentage")

# Plot the discounts
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Highest discounted products
ax[0].barh(top_discounted["product_name"], top_discounted["discount_percentage"], color="blue")
ax[0].set_title("Top 5 Highest Discounted Products")
ax[0].set_xlabel("Discount Percentage")
ax[0].invert_yaxis()

# Lowest discounted products
ax[1].barh(lowest_discounted["product_name"], lowest_discounted["discount_percentage"], color="red")
ax[1].set_title("Top 5 Lowest Discounted Products")
ax[1].set_xlabel("Discount Percentage")
ax[1].invert_yaxis()

plt.tight_layout()
plt.show()
