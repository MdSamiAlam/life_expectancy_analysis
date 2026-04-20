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

# Missing Values

print("\nMissing Values per Column\n")
print(df.isnull().sum())

# Handling Missing Values

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



# Normality Check


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



# Outlier Detection (IQR)

print("\nHandling Outliers using IQR Method\n")

Q1 = df["Life expectancy"].quantile(0.25)
Q3 = df["Life expectancy"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", round(lower_bound,2))
print("Upper Bound:", round(upper_bound,2))



# Outlier Treatment (Capping)

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



# Correlation Analysis

print("\n Correlation with Life Expectancy\n")

num_df = df.select_dtypes(include=np.number)

corr_matrix = num_df.corr()

print("\nCorrelation Matrix (rounded):\n")
print(corr_matrix.round(2))

corr_with_le = corr_matrix["Life expectancy"].drop("Life expectancy")

print("\nTop Correlations:\n")
print(corr_with_le.sort_values(ascending=False).round(3))


# SECTION 2 — DATA VISUALISATIONS


# Objective 3:
# Visualise the relationship between Adult Mortality rate and Life Expectancy
# using a scatter plot to observe direction and strength.

# Scatter Plot with regression line
plt.figure(figsize=(8,5))
sns.regplot(data=df, x="Adult Mortality", y="Life expectancy",
            scatter_kws={"alpha":0.4},
            line_kws={"color":"red"})
plt.title("Adult Mortality vs Life Expectancy")
plt.xlabel("Adult Mortality")
plt.ylabel("Life Expectancy")
plt.tight_layout()
plt.show()

print("➡ Strong negative relationship observed")




# Objective 4:
# Compare the distribution of Life Expectancy between Developed and
# Developing countries using a box plot.

# Boxplot
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="Status", y="Life expectancy")
plt.title("Life Expectancy by Country Status")
plt.tight_layout()
plt.show()

print("➡ Developed countries show higher life expectancy")




# Objective 5:
# Identify important health and economic indicators that correlate
# with Life Expectancy using a correlation heatmap.

# Heatmap (selected features)
cols = ["Life expectancy","Schooling","GDP","Adult Mortality","HIV/AIDS"]
plt.figure(figsize=(8,6))
sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Key Features)")
plt.tight_layout()
plt.show()

# SECTION 3 — LINEAR REGRESSION


# Objective 1:
# Analyse the impact of Schooling on Life Expectancy using linear regression

# Objective 2:
# Analyse the impact of GDP on Life Expectancy using linear regression
# (log transformation applied due to skewness)

def regression_model(x_col, display_name=None):
    display_name = display_name or x_col
    
    data = df[[x_col, "Life expectancy"]].dropna()
    
    X = data[[x_col]].values
    y = data["Life expectancy"].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\n{display_name} → Life Expectancy")
    print("-"*40)
    print("Slope:", round(model.coef_[0],4))
    print("Intercept:", round(model.intercept_,4))
    print("R²:", round(r2,4))
    print("RMSE:", round(rmse,4))
    
    # Sort for proper line
    x_sorted = np.sort(X, axis=0)
    
    plt.figure(figsize=(7,5))
    plt.scatter(X, y, alpha=0.3)
    plt.plot(x_sorted, model.predict(x_sorted), color="red")
    plt.title(f"{display_name} vs Life Expectancy")
    plt.xlabel(display_name)
    plt.ylabel("Life Expectancy")
    plt.tight_layout()
    plt.show()
    
    return model.coef_[0]


# 🔹 Model 1 — Schooling
s1 = regression_model("Schooling", "Schooling")
print(f"➡ 1 year increase in schooling → {s1:.2f} years increase in life expectancy")


# 🔹 Model 2 — GDP (FIXED using log)

# Ensure no zero or negative values (safety)
df = df[df["GDP"] > 0]

# Log transform
df["log_GDP"] = np.log1p(df["GDP"])

s2 = regression_model("log_GDP", "Log(GDP)")
print(f"➡ Log(GDP) increase → {s2:.2f} increase in life expectancy")

print("\nNote: GDP was skewed, so log transformation was applied for better model fit.")

# Hypothesis test

print("\n" + "="*50)
print("SECTION 4 — HYPOTHESIS TEST")
print("="*50)

developed = df[df["Status"]=="Developed"]["Life expectancy"]
developing = df[df["Status"]=="Developing"]["Life expectancy"]

t_stat, p_val = stats.ttest_ind(developed, developing,
                                equal_var=False,
                                alternative="greater")

print("Mean Developed:", round(developed.mean(),2))
print("Mean Developing:", round(developing.mean(),2))
print("T-stat:", round(t_stat,4))
print("P-value:", p_val)

if p_val < 0.05:
    print(" Reject H0 → Developed countries have higher life expectancy")
else:
    print("❌ Fail to reject H0")

# Visualization for Hypothesis Test

means = [developed.mean(), developing.mean()]
labels = ["Developed", "Developing"]

bars = plt.bar(labels, means, width=0.4) 

# Add mean values on top
for bar, val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{val:.1f}",
             ha="center")

plt.title("Life Expectancy Comparison")
plt.ylabel("Life Expectancy")

plt.tight_layout()
plt.savefig("hypothesis_ttest_comparison.png")
plt.show()
