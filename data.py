
# ============================================================
# DATA ANALYSIS ON WHO LIFE EXPECTANCY DATASET
# Dataset: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SETTINGS
# ============================================================

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

TARGET = "Life expectancy"

# Project-defined categories (not official WHO categories)
# Low = below 60
# Medium = 60 to below 75
# High = 75 or above

CLASS_BINS = [-np.inf, 60, 75, np.inf]
CLASS_LABELS = ["Low", "Medium", "High"]


# ============================================================
# SECTION 1 — LOAD AND INSPECT DATA
# ============================================================

df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()

print("Dataset loaded:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe().round(2))

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Remove rows where target is missing
df = df.dropna(subset=[TARGET]).copy()

print("\nDataset shape after cleaning:", df.shape)


# ============================================================
# SECTION 2 — HANDLE MISSING VALUES FOR EDA
# ============================================================

eda_df = df.copy()

# Fill numerical missing values using median
for col in eda_df.select_dtypes(include=np.number).columns:
    eda_df[col] = eda_df[col].fillna(eda_df[col].median())

# Fill categorical missing values using mode
for col in eda_df.select_dtypes(exclude=np.number).columns:
    if eda_df[col].notna().any():
        eda_df[col] = eda_df[col].fillna(eda_df[col].mode()[0])

print("\nMissing values after EDA cleaning:")
print(eda_df.isnull().sum())


# ============================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 50)
print("SECTION 3 — EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 1. Histogram of Life Expectancy
plt.figure(figsize=(8, 5))
sns.histplot(eda_df[TARGET], kde=True)
plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy (years)")
plt.tight_layout()
plt.show()

# 2. Boxplot of Life Expectancy
plt.figure(figsize=(8, 4))
sns.boxplot(x=eda_df[TARGET])
plt.title("Life Expectancy Boxplot")
plt.tight_layout()
plt.show()

# 3. Correlation with Life Expectancy
numeric_df = eda_df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

print("\nCorrelation with Life Expectancy:")
print(
    correlation[TARGET]
    .drop(TARGET)
    .sort_values(ascending=False)
    .round(3)
)

# 4. Correlation Heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7}
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. GDP Groups vs Average Life Expectancy
if "GDP" in eda_df.columns:

    eda_df["GDP Group"] = pd.qcut(
        eda_df["GDP"].rank(method="first"),
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=eda_df,
        x="GDP Group",
        y=TARGET,
        errorbar=None
    )

    plt.title("Average Life Expectancy by GDP Group")
    plt.xlabel("GDP Group")
    plt.ylabel("Average Life Expectancy")
    plt.tight_layout()
    plt.show()

# 6. Developed vs Developing Countries
if "Status" in eda_df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=eda_df,
        x="Status",
        y=TARGET
    )

    plt.title("Life Expectancy by Country Status")
    plt.tight_layout()
    plt.show()


# ============================================================
# SECTION 4 — LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4 — LINEAR REGRESSION")
print("=" * 50)

data = df[["Schooling", TARGET]].dropna()

X = data[["Schooling"]]
y = data[TARGET]

# Split by country to prevent country overlap
groups = df.loc[data.index, "Country"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\nSchooling vs Life Expectancy")
print("Slope:", round(model.coef_[0], 4))
print("Intercept:", round(model.intercept_, 4))
print("Test R2:", round(r2_score(y_test, prediction), 4))

print(
    "Test RMSE:",
    round(np.sqrt(mean_squared_error(y_test, prediction)), 4)
)

# Regression Line
order = np.argsort(X_test["Schooling"].values)

plt.figure(figsize=(7, 5))

plt.scatter(
    X_test["Schooling"],
    y_test,
    alpha=0.3,
    label="Actual"
)

plt.plot(
    X_test["Schooling"].values[order],
    prediction[order],
    color="red",
    label="Regression line"
)

plt.title("Schooling vs Life Expectancy")
plt.xlabel("Schooling (years)")
plt.ylabel("Life Expectancy (years)")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 5 — PREPARE MACHINE-LEARNING DATA
# ============================================================

print("\n" + "=" * 50)
print("SECTION 5 — MACHINE-LEARNING PREPARATION")
print("=" * 50)

ml_df = df.copy()

# Convert numerical life expectancy into three classes
ml_df["Life Expectancy Class"] = pd.cut(
    ml_df[TARGET],
    bins=CLASS_BINS,
    labels=CLASS_LABELS,
    right=False
)

# Keep Country separately for group splitting
groups = ml_df["Country"]

# Remove target and Country from model features
columns_to_drop = [
    TARGET,
    "Life Expectancy Class",
    "Country"
]

X = ml_df.drop(
    columns=[
        col for col in columns_to_drop
        if col in ml_df.columns
    ]
)

y = ml_df["Life Expectancy Class"].astype(str)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


# ============================================================
# FEATURE SCALING AND PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# SECTION 6 — DEFINE FOUR MACHINE-LEARNING MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

# Train/test split configurations
splits = {
    "80:20": 0.20,
    "70:30": 0.30,
    "75:25": 0.25
}

results = []
cv_results = []


# ============================================================
# SECTION 7 — TRAIN, TEST AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 50)
print("SECTION 7 — MODEL TRAINING AND EVALUATION")
print("=" * 50)

for split_name, test_size in splits.items():

    print("\n" + "-" * 50)
    print("Train/Test Split:", split_name)
    print("-" * 50)

    # Country-grouped split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42
    )

    train_index, test_index = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    groups_train = groups.iloc[train_index]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print(
        "Actual test percentage:",
        round(len(X_test) / len(X) * 100, 2),
        "%"
    )

    train_countries = set(groups.iloc[train_index])
    test_countries = set(groups.iloc[test_index])

    print(
        "Countries appearing in both sets:",
        len(train_countries.intersection(test_countries))
    )

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        prediction = pipeline.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, prediction)

        precision = precision_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "Split": split_name,
            "Training Rows": len(X_train),
            "Testing Rows": len(X_test),
            "Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1
        })

        print(
            f"\n{model_name}: "
            f"Accuracy={accuracy:.3f}, "
            f"Precision={precision:.3f}, "
            f"Recall={recall:.3f}, "
            f"Macro F1={f1:.3f}"
        )

        # Group-aware five-fold cross-validation
        smallest_class = int(y_train.value_counts().min())
        number_of_countries = groups_train.nunique()

        folds = min(
            5,
            smallest_class,
            number_of_countries
        )

        if folds >= 2:

            cv = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                groups=groups_train,
                cv=cv,
                scoring="f1_macro"
            )

            cv_results.append({
                "Model": model_name,
                "Split": split_name,
                "CV Folds": folds,
                "Mean CV Macro F1": scores.mean(),
                "CV F1 Std": scores.std()
            })

        # Confusion Matrix
        cm = confusion_matrix(
            y_test,
            prediction,
            labels=CLASS_LABELS
        )

        print(
            f"\nConfusion Matrix — {model_name} ({split_name})"
        )

        print("Order: Low, Medium, High")
        print(cm)

        # Confusion Matrix Visualization
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_LABELS
        )

        display.plot(
            cmap="Blues",
            values_format="d"
        )

        plt.title(f"{model_name} — {split_name} Split")
        plt.tight_layout()
        plt.show()


# ============================================================
# SECTION 7.1 — DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 50)
print("DECISION TREE VISUALIZATION")
print("=" * 50)

# Use the same 80:20 country-grouped split
tree_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

tree_train_idx, tree_test_idx = next(
    tree_splitter.split(X, y, groups=groups)
)

tree_X_train = X.iloc[tree_train_idx]
tree_y_train = y.iloc[tree_train_idx]

# Train Decision Tree
tree_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

tree_pipeline.fit(tree_X_train, tree_y_train)

# Get feature names
feature_names = (
    tree_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

# Plot first three levels
plt.figure(figsize=(24, 12))

plot_tree(
    tree_pipeline.named_steps["model"],
    feature_names=feature_names,
    class_names=tree_pipeline.named_steps["model"].classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title(
    "Decision Tree Visualization - Life Expectancy Classification",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "decision_tree_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Decision Tree visualization saved successfully.")


# ============================================================
# SECTION 8 — FUTURE-YEAR VALIDATION
# ============================================================

print("\n" + "=" * 50)
print("SECTION 8 — FUTURE-YEAR VALIDATION")
print("=" * 50)

future_results_df = pd.DataFrame()

if "Year" in ml_df.columns and ml_df["Year"].nunique() >= 4:

    # Use last three available years for testing
    years = sorted(ml_df["Year"].dropna().unique())

    future_start_year = years[-3]

    earlier_data = ml_df[
        ml_df["Year"] < future_start_year
    ].copy()

    future_data = ml_df[
        ml_df["Year"] >= future_start_year
    ].copy()

    if not earlier_data.empty and not future_data.empty:

        future_X_train = earlier_data.drop(
            columns=[
                col for col in columns_to_drop
                if col in earlier_data.columns
            ]
        )

        future_X_test = future_data.drop(
            columns=[
                col for col in columns_to_drop
                if col in future_data.columns
            ]
        )

        future_y_train = earlier_data[
            "Life Expectancy Class"
        ].astype(str)

        future_y_test = future_data[
            "Life Expectancy Class"
        ].astype(str)

        print("Training years: before", future_start_year)
        print("Future test years:", future_start_year, "onwards")
        print("Training rows:", len(future_X_train))
        print("Future test rows:", len(future_X_test))

        future_results = []

        for model_name, model in models.items():

            future_pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            future_pipeline.fit(
                future_X_train,
                future_y_train
            )

            future_prediction = future_pipeline.predict(
                future_X_test
            )

            future_results.append({
                "Model": model_name,

                "Accuracy": accuracy_score(
                    future_y_test,
                    future_prediction
                ),

                "Macro Precision": precision_score(
                    future_y_test,
                    future_prediction,
                    average="macro",
                    zero_division=0
                ),

                "Macro Recall": recall_score(
                    future_y_test,
                    future_prediction,
                    average="macro",
                    zero_division=0
                ),

                "Macro F1": f1_score(
                    future_y_test,
                    future_prediction,
                    average="macro",
                    zero_division=0
                )
            })

        future_results_df = pd.DataFrame(future_results)

        print("\nFuture-Year Results:")

        print(
            future_results_df
            .round(4)
            .to_string(index=False)
        )

        future_results_df.to_csv(
            "future_year_results.csv",
            index=False
        )

        print("Saved: future_year_results.csv")

    else:
        print("Not enough data for future-year validation.")

else:
    print("Year column unavailable or contains too few unique years.")


# ============================================================
# SECTION 9 — COMPARE MODELS AND DECLARE THE WINNER
# ============================================================

print("\n" + "=" * 60)
print("SECTION 9 — FINAL MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)

print("\nResults from all 12 experiments:")

print(
    results_df
    .round(4)
    .to_string(index=False)
)

print("\nGroup-aware cross-validation results:")

if not cv_df.empty:

    print(
        cv_df
        .round(4)
        .to_string(index=False)
    )

else:
    print("Cross-validation was not available for these splits.")

# Save results
results_df.to_csv(
    "ml_experiment_results.csv",
    index=False
)

cv_df.to_csv(
    "ml_cross_validation_results.csv",
    index=False
)

print("\nSaved:")
print("1. ml_experiment_results.csv")
print("2. ml_cross_validation_results.csv")

# Macro F1 comparison
plt.figure(figsize=(11, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="Macro F1",
    hue="Split",
    errorbar=None
)

plt.title("Macro F1 Comparison — Models and Splits")
plt.ylabel("Macro F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Average model performance
model_summary = (
    results_df.groupby("Model")[
        [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ]
    ]
    .mean()
    .sort_values("Macro F1", ascending=False)
)

print("\nAverage model performance:")

print(
    model_summary
    .round(4)
    .to_string()
)

# Select best model using average Macro F1
best_model = model_summary.index[0]

best_f1 = model_summary.iloc[0]["Macro F1"]

best_accuracy = model_summary.loc[
    best_model,
    "Accuracy"
]


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)

print(
    f"{best_model} achieved the highest average Macro F1 "
    f"of {best_f1:.4f} across the three country-grouped experiments."
)

print(
    f"Its average accuracy across these experiments was "
    f"{best_accuracy:.4f}."
)

print(
    f"Therefore, {best_model} is the best-performing model "
    "according to average Macro F1 in this evaluation."
)

print(
    "\nThis conclusion is based on the current dataset and evaluation setup. "
    "The separate future-year results should also be reviewed before making "
    "claims about future predictions."
)

print(
    "\nThe classification boundaries are project-defined: "
    "Low < 60, Medium = 60 to below 75, and High >= 75. "
    "They are not official WHO categories."
)

print("\nAnalysis completed successfully.")
