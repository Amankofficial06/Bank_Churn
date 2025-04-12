import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\amand\Downloads\Bank_churn.csv")

# Statistical summary
print("Basic Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Check for null values
print("\nMissing Values:")
print(df.isnull().sum())

# Correlation matrix
correlation = df.corr(numeric_only=True)

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['CreditScore'], bins=30, kde=True, color='cornflowerblue', edgecolor='black')
plt.title('Histogram of CreditScore')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Churn count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Exited')
plt.title('Churn Count')
plt.xlabel('Exited (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()

# Distribution of Age
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot of Age vs. Exited
plt.figure(figsize=(8, 5))
sns.boxplot(x='Exited', y='Age', data=df)
plt.title('Age vs. Churn')
plt.show()

# Balance distribution by churn status
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='Balance', hue='Exited', fill=True)
plt.title('Balance Distribution by Churn Status')
plt.show()

# Gender vs Churn
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender', hue='Exited')
plt.title('Gender vs Churn')
plt.show()

# Geography vs Churn
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Geography', hue='Exited')
plt.title('Geography vs Churn')
plt.show()



# Scatter plot: Age vs Estimated Salary, colored by Exited
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='EstimatedSalary', hue='Exited', palette='coolwarm', alpha=0.6)
plt.title('Scatter Plot: Age vs Estimated Salary (Churn Highlighted)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.grid(True)
plt.legend(title='Exited (Churned)')
plt.show()



# Function to detect outliers using IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Example: Detect outliers in 'CreditScore'
outliers_credit = detect_outliers_iqr(df, 'CreditScore')
print(f"Outliers in CreditScore: {len(outliers_credit)} rows")
print(outliers_credit[['CustomerId', 'CreditScore']])




# Group by Age and calculate mean CreditScore
age_credit = df.groupby('Age')['CreditScore'].mean().reset_index()

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(age_credit['Age'], age_credit['CreditScore'], marker='o', color='teal')
plt.title('Average Credit Score by Age')
plt.xlabel('Age')
plt.ylabel('Average Credit Score')
plt.grid(True)
plt.show()




