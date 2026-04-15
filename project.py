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
