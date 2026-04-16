# Dataset : (Kaggle):
# https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# Global style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# Load Data
df = pd.read_csv("Life Expectancy Data.csv")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Show dataset shape
print("Dataset loaded:", df.shape, "\n")

# Project Objectives
objectives = """
Objective 1 (Linear Regression):
Analyse the impact of Schooling on Life Expectancy.
X = Schooling   |   Y = Life expectancy

Objective 2 (Linear Regression):
Examine GDP vs Life Expectancy.
X = GDP   |   Y = Life expectancy

Objective 3 (Visualisation - Scatter Plot):
Adult Mortality vs Life Expectancy.

Objective 4 (Visualisation - Box Plot):
Life Expectancy by country status.

Objective 5 (Visualisation - Heatmap):
Correlation between numerical features.
"""
print(objectives)

# BASIC EDA

# First few rows
print(df.head())

# Dataset info
print("\nDataset Info\n")
df.info()

# Summary statistics
print("\nSummary Statistics\n")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# Missing Values
# ─────────────────────────────────────────────
print("\nMissing Values per Column\n")
print(df.isnull().sum())


# ─────────────────────────────────────────────
# Handling Missing Values
# ─────────────────────────────────────────────

# LOW missing → mean
low_cols = ["Life expectancy","Adult Mortality","BMI","Polio","Diphtheria"]
for col in low_cols:
    df[col] = df[col].fillna(df[col].mean())

# MEDIUM missing → group mean
medium_cols = ["Alcohol","Total expenditure","Schooling","Income composition of resources"]
for col in medium_cols:
    df[col] = df.groupby("Status")[col].transform(lambda x: x.fillna(x.mean()))

# HIGH missing → drop
df.drop(columns=["Population"], inplace=True)

# Remaining important columns → mean
df["GDP"] = df["GDP"].fillna(df["GDP"].mean())
df["Hepatitis B"] = df["Hepatitis B"].fillna(df["Hepatitis B"].mean())

print("\nMissing Values after handling\n")
print(df.isnull().sum())


# ─────────────────────────────────────────────
# Normality Check
# ─────────────────────────────────────────────

# Histogram
plt.figure()
sns.histplot(df["Life expectancy"], kde=True)
plt.title("Histogram of Life Expectancy")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x=df["Life expectancy"])
plt.title("Boxplot of Life Expectancy")
plt.show()

print("\nObservation: Data is negatively skewed (not normal)")


# ─────────────────────────────────────────────
# Outlier Detection (IQR)
# ─────────────────────────────────────────────
print("\nHandling Outliers using IQR Method\n")

Q1 = df["Life expectancy"].quantile(0.25)
Q3 = df["Life expectancy"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", round(lower_bound,2))
print("Upper Bound:", round(upper_bound,2))


# ─────────────────────────────────────────────
# Outlier Treatment (Capping)
# ─────────────────────────────────────────────
df["Life expectancy"] = np.where(
    df["Life expectancy"] < lower_bound,
    lower_bound,
    df["Life expectancy"]
)

df["Life expectancy"] = np.where(
    df["Life expectancy"] > upper_bound,
    upper_bound,
    df["Life expectancy"]
)

print("\nOutliers handled using capping")
print("Dataset shape:", df.shape)


# Check after treatment
plt.figure()
sns.boxplot(x=df["Life expectancy"])
plt.title("Boxplot after Outlier Treatment")
plt.show()


# ─────────────────────────────────────────────
# Correlation Analysis
# ─────────────────────────────────────────────
print("\n Correlation with Life Expectancy\n")

num_df = df.select_dtypes(include=np.number)

corr_matrix = num_df.corr()

print("\nCorrelation Matrix (rounded):\n")
print(corr_matrix.round(2))

corr_with_le = corr_matrix["Life expectancy"].drop("Life expectancy")

print("\nTop Correlations:\n")
print(corr_with_le.sort_values(ascending=False).round(3))
