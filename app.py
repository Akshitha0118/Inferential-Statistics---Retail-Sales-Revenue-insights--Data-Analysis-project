import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("Sales Data Analysis for Retail Store")
st.write("This application analyzes sales data for various product categories.")

# Generate synthetic sales data
def generate_data():
    np.random.seed(42)
    data = {
        'product_id': range(1, 21),
        'product_name': [f'Product {i}' for i in range(1, 21)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 20),
        'units_sold': np.random.poisson(lam=20, size=20),
        'sale_date': pd.date_range(start='2023-01-01', periods=20, freq='D')
    }
    return pd.DataFrame(data)

sales_data = generate_data()

# Display data
st.subheader("Sales Data")
st.dataframe(sales_data)

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.write(sales_data['units_sold'].describe())

mean_sales = sales_data['units_sold'].mean()
median_sales = sales_data['units_sold'].median()
mode_sales = sales_data['units_sold'].mode()[0]

st.write(f"Mean Units Sold: {mean_sales:.2f}")
st.write(f"Median Units Sold: {median_sales}")
st.write(f"Mode Units Sold: {mode_sales}")

# Category-wise statistics
category_stats = (
    sales_data
    .groupby('category')['units_sold']
    .agg(['sum', 'mean', 'std'])
    .reset_index()
)

category_stats.columns = [
    'Category',
    'Total Units Sold',
    'Average Units Sold',
    'Std Dev of Units Sold'
]

st.subheader("Category Statistics")
st.dataframe(category_stats)

# Inferential statistics
confidence_level = 0.95
df = len(sales_data) - 1
std_error = sales_data['units_sold'].std() / np.sqrt(len(sales_data))
t_score = stats.t.ppf((1 + confidence_level) / 2, df)

margin_error = t_score * std_error
confidence_interval = (
    mean_sales - margin_error,
    mean_sales + margin_error
)

st.subheader("Confidence Interval for Mean Units Sold")
st.write(confidence_interval)

# Hypothesis testing
t_stat, p_value = stats.ttest_1samp(sales_data['units_sold'], 20)

st.subheader("Hypothesis Testing (One-Sample t-test)")
st.write(f"T-statistic: {t_stat:.4f}")
st.write(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    st.success("Reject the null hypothesis: Mean units sold ≠ 20")
else:
    st.info("Fail to reject the null hypothesis: Mean units sold ≈ 20")

# -------------------- VISUALIZATIONS --------------------
st.subheader("Visualizations")

# Histogram
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(sales_data['units_sold'], bins=10, kde=True, ax=ax1)
ax1.axvline(mean_sales, linestyle='--', label='Mean')
ax1.axvline(median_sales, linestyle='--', label='Median')
ax1.axvline(mode_sales, linestyle='--', label='Mode')
ax1.set_title("Distribution of Units Sold")
ax1.set_xlabel("Units Sold")
ax1.set_ylabel("Frequency")
ax1.legend()
st.pyplot(fig1)

# Boxplot
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.boxplot(x='category', y='units_sold', data=sales_data, ax=ax2)
ax2.set_title("Boxplot of Units Sold by Category")
ax2.set_xlabel("Category")
ax2.set_ylabel("Units Sold")
st.pyplot(fig2)

# Bar plot
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(
    x='Category',
    y='Total Units Sold',
    data=category_stats,
    ax=ax3
)
ax3.set_title("Total Units Sold by Category")
ax3.set_xlabel("Category")
ax3.set_ylabel("Total Units Sold")
st.pyplot(fig3)
