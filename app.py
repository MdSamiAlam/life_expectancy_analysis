# ============================================================
# Life Expectancy Analysis & Machine Learning Dashboard
# ============================================================

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ------------------------------------------------------------
# PAGE CONFIGURATION AND STYLE
# ------------------------------------------------------------

st.set_page_config(
    page_title="Life Expectancy Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Life Expectancy Data.csv"

EXPERIMENT_FILE = BASE_DIR / "ml_experiment_results.csv"
CV_FILE = BASE_DIR / "ml_cross_validation_results.csv"
FUTURE_FILE = BASE_DIR / "future_year_results.csv"
TREE_IMAGE = BASE_DIR / "decision_tree_visualization.png"

TARGET = "Life expectancy"
CLASS_LABELS = ["Low", "Medium", "High"]


# ------------------------------------------------------------
# DATA LOADING AND CLEANING — MATCHES CURRENT data.py
# ------------------------------------------------------------

@st.cache_data
def load_data(file_path, modified_time):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    original_shape = data.shape
    duplicate_count = int(data.duplicated().sum())

    # Same initial cleaning as data.py
    data = data.drop_duplicates().copy()
    data = data.dropna(subset=[TARGET]).copy()

    # Same EDA missing-value treatment as data.py
    eda_data = data.copy()

    for col in eda_data.select_dtypes(include=np.number).columns:
        eda_data[col] = eda_data[col].fillna(eda_data[col].median())

    for col in eda_data.select_dtypes(exclude=np.number).columns:
        if eda_data[col].notna().any():
            modes = eda_data[col].mode()
            if not modes.empty:
                eda_data[col] = eda_data[col].fillna(modes[0])

    return data, eda_data, original_shape, duplicate_count


def load_optional_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")
    return pd.DataFrame()


def show_figure(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


if not DATA_FILE.exists():
    st.error(
        "`Life Expectancy Data.csv` was not found. "
        "Place it in the same folder as `app.py`."
    )
    st.stop()

try:
    df, eda_df, original_shape, duplicate_count = load_data(
        str(DATA_FILE), DATA_FILE.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Unable to load or clean the dataset: {exc}")
    st.stop()


# ------------------------------------------------------------
# LOAD MODEL OUTPUT FILES CREATED BY data.py
# ------------------------------------------------------------

results_df = load_optional_csv(EXPERIMENT_FILE)
cv_df = load_optional_csv(CV_FILE)
future_df = load_optional_csv(FUTURE_FILE)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("🌍 Life Expectancy Analysis Dashboard")
st.markdown(
    "**Dataset:** WHO Life Expectancy (2000–2015)  |  "
    "**Source:** [Kaggle – WHO Life Expectancy]"
    "(https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)"
)
st.caption(
    "Exploratory data analysis, regression analysis and "
    "machine-learning classification."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Original rows", f"{original_shape[0]:,}")
m2.metric("Rows after cleaning", f"{len(df):,}")
m3.metric("Columns", f"{df.shape[1]}")
m4.metric("Duplicate rows removed", f"{duplicate_count:,}")

st.markdown("---")


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------

SECTIONS = [
    "Project Overview",
    "Data Preview",
    "Dataset Information",
    "Summary Statistics",
    "Missing Values",
    "Life Expectancy Distribution",
    "Correlation Analysis",
    "GDP Group Analysis",
    "Developed vs Developing",
    "Regression Analysis",
    "Classification Overview",
    "Model Comparison",
    "Cross-Validation Results",
    "Future-Year Validation",
    "Decision Tree Visualization"
]

st.sidebar.title("Navigation")
section = st.sidebar.radio("Select a section:", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset after initial cleaning**")
st.sidebar.write(f"{df.shape[0]:,} rows × {df.shape[1]} columns")


# ------------------------------------------------------------
# 1. PROJECT OVERVIEW
# ------------------------------------------------------------

if section == "Project Overview":
    st.header("Project Overview")
    st.caption("Analysis of WHO country-year data using EDA, regression and classification.")
    st.write(
        f"**Dataset snapshot:** {len(df):,} usable rows and {df.shape[1]} columns "
        f"after initial cleaning. The dashboard starts with {original_shape[0]:,} "
        f"rows and removed {duplicate_count:,} duplicate rows."
    )


    st.subheader("Project Objectives")
    objectives = [
        "Inspect and clean the WHO Life Expectancy dataset.",
        "Explore distributions, missing values and relationships among variables.",
        "Study the relationship between Schooling and Life expectancy using linear regression.",
        "Compare life expectancy across GDP groups and country development status.",
        "Classify life expectancy into Low, Medium and High categories.",
        "Compare Decision Tree, KNN, Gaussian Naive Bayes and Random Forest.",
        "Evaluate models using multiple train-test ratios, Macro F1 and group-aware cross-validation.",
        "Examine performance on the last three available years."
    ]
    for item in objectives:
        st.markdown(f"- {item}")

    st.subheader("Machine-Learning Target Categories")
    category_table = pd.DataFrame({
        "Class": ["Low", "Medium", "High"],
        "Life expectancy range": [
            "Below 60 years",
            "60 to below 75 years",
            "75 years or above"
        ]
    })
    st.dataframe(category_table, hide_index=True, use_container_width=True)
    st.info(
        "These class boundaries are project-defined thresholds; "
        "they are not official WHO categories."
    )

    st.subheader("Algorithms")
    st.markdown(
        "- Decision Tree\n"
        "- K-Nearest Neighbors (KNN)\n"
        "- Gaussian Naive Bayes\n"
        "- Random Forest (200 trees)"
    )


# ------------------------------------------------------------
# 2. DATA PREVIEW
# ------------------------------------------------------------

elif section == "Data Preview":
    st.header("Data Preview")
    st.caption(
        "A sample of the cleaned dataset. Each row represents one country in one year."
    )

    # Compact dataset coverage indicators
    preview_col1, preview_col2, preview_col3 = st.columns(3)
    preview_col1.metric("Countries", f"{df['Country'].nunique():,}" if "Country" in df.columns else "N/A")
    if "Year" in df.columns:
        preview_col2.metric("Year range", f"{int(df['Year'].min())}–{int(df['Year'].max())}")
    else:
        preview_col2.metric("Year range", "N/A")
    preview_col3.metric("Rows after cleaning", f"{len(df):,}")

    st.info(
        "Review country names, years, target values and predictor columns for inconsistencies."
    )
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Names")
    st.write(list(df.columns))


# ------------------------------------------------------------
# 3. DATASET INFORMATION
# ------------------------------------------------------------

elif section == "Dataset Information":
    st.header("Dataset Information")
    st.caption("Dataset dimensions, data types and feature groups.")
    st.write(
        f"**Column-type summary:** "
        f"{len(df.select_dtypes(include=np.number).columns)} numerical columns and "
        f"{len(df.select_dtypes(exclude=np.number).columns)} categorical columns."
    )


    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue(), language="text")

    st.write("**Numerical columns:**")
    st.write(df.select_dtypes(include=np.number).columns.tolist())

    st.write("**Categorical columns:**")
    st.write(df.select_dtypes(exclude=np.number).columns.tolist())


# ------------------------------------------------------------
# 4. SUMMARY STATISTICS
# ------------------------------------------------------------

elif section == "Summary Statistics":
    st.header("Summary Statistics")
    st.caption("Descriptive statistics for the EDA dataset.")
    st.write(
        f"**Life expectancy in this dataset:** mean "
        f"{eda_df[TARGET].mean():.2f} years; median "
        f"{eda_df[TARGET].median():.2f} years; observed range "
        f"{eda_df[TARGET].min():.2f}–{eda_df[TARGET].max():.2f} years."
    )

    st.dataframe(
        eda_df.describe().round(2),
        use_container_width=True
    )


# ------------------------------------------------------------
# 5. MISSING VALUES
# ------------------------------------------------------------

elif section == "Missing Values":
    st.header("Missing Values")
    st.caption("Missing-value counts before and after EDA imputation.")


    before = df.isnull().sum().rename("Missing Before EDA Imputation")
    after = eda_df.isnull().sum().rename("Missing After EDA Imputation")

    missing_table = pd.concat([before, after], axis=1).reset_index()
    missing_table = missing_table.rename(columns={"index": "Column"})
    st.dataframe(missing_table, use_container_width=True, hide_index=True)

    st.subheader("Missing-Value Treatment")
    st.markdown(
        """
        - Duplicate rows are removed.
        - Rows with a missing `Life expectancy` target are removed.
        - Missing numerical values for EDA are filled using the column median.
        - Missing categorical values for EDA are filled using the column mode.
        - In the machine-learning pipeline, numerical missing values are
          imputed using the median and categorical missing values using
          the most frequent value. These imputers are fitted within the
          training pipeline.
        """
    )


# ------------------------------------------------------------
# 6. LIFE EXPECTANCY DISTRIBUTION
# ------------------------------------------------------------

elif section == "Life Expectancy Distribution":
    st.header("Life Expectancy Distribution")
    st.caption("Distribution and spread of observed life-expectancy values.")
    st.write(
        f"**Observed summary:** mean {eda_df[TARGET].mean():.2f} years; "
        f"median {eda_df[TARGET].median():.2f} years; "
        f"standard deviation {eda_df[TARGET].std():.2f} years."
    )


    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(eda_df[TARGET], kde=True, ax=ax)
        ax.set_title("Distribution of Life Expectancy")
        ax.set_xlabel("Life expectancy (years)")
        show_figure(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=eda_df[TARGET], ax=ax)
        ax.set_title("Life Expectancy Boxplot")
        ax.set_xlabel("Life expectancy (years)")
        show_figure(fig)

    st.caption(
        "These charts use the EDA-cleaned data. The current data.py "
        "does not apply IQR capping to the life expectancy target."
    )


# ------------------------------------------------------------
# 7. CORRELATION ANALYSIS
# ------------------------------------------------------------

elif section == "Correlation Analysis":
    st.header("Correlation Analysis")
    st.caption("Linear associations between numerical variables.")
    corr_for_takeaway = eda_df.select_dtypes(include=np.number).corr()[TARGET].drop(TARGET).dropna()
    if not corr_for_takeaway.empty:
        strongest_pos = corr_for_takeaway.idxmax()
        strongest_neg = corr_for_takeaway.idxmin()
        st.write(
            f"**Strongest positive association:** {strongest_pos} "
            f"(r = {corr_for_takeaway[strongest_pos]:.3f}).  "
            f"**Strongest negative association:** {strongest_neg} "
            f"(r = {corr_for_takeaway[strongest_neg]:.3f})."
        )


    numeric_df = eda_df.select_dtypes(include=np.number)
    correlation = numeric_df.corr()

    if TARGET in correlation.columns:
        corr_target = (
            correlation[TARGET]
            .drop(TARGET)
            .sort_values(ascending=False)
            .rename("Correlation with Life expectancy")
            .reset_index()
            .rename(columns={"index": "Feature"})
        )

        st.subheader("Correlation with Life Expectancy")
        st.dataframe(
            corr_target.round(3),
            hide_index=True,
            use_container_width=True
        )

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        ax=ax
    )
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    show_figure(fig)


# ------------------------------------------------------------
# 8. GDP GROUP ANALYSIS
# ------------------------------------------------------------

elif section == "GDP Group Analysis":
    st.header("GDP Groups vs Average Life Expectancy")
    st.caption("Average life expectancy across five GDP groups.")


    if "GDP" not in eda_df.columns:
        st.warning("GDP column is not available.")
    else:
        plot_df = eda_df[["GDP", TARGET]].dropna().copy()
        try:
            plot_df["GDP Group"] = pd.qcut(
                plot_df["GDP"].rank(method="first"),
                q=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"]
            )

            summary = (
                plot_df.groupby("GDP Group", observed=False)[TARGET]
                .mean()
                .reset_index()
            )
            st.dataframe(
                summary.rename(columns={TARGET: "Average Life expectancy"})
                .round(2),
                hide_index=True,
                use_container_width=True
            )

            if not summary.empty:
                highest_group = summary.loc[summary[TARGET].idxmax()]
                lowest_group = summary.loc[summary[TARGET].idxmin()]
                difference = highest_group[TARGET] - lowest_group[TARGET]
                st.info(
                    f"**Result from this dataset:** the {highest_group['GDP Group']} "
                    f"GDP group has the highest average life expectancy "
                    f"({highest_group[TARGET]:.2f} years), while the "
                    f"{lowest_group['GDP Group']} group has the lowest "
                    f"({lowest_group[TARGET]:.2f} years). The difference is "
                    f"{difference:.2f} years. This is an association, not a causal effect."
                )

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(
                data=plot_df,
                x="GDP Group",
                y=TARGET,
                errorbar=None,
                ax=ax
            )
            ax.set_title("Average Life Expectancy by GDP Group")
            ax.set_xlabel("GDP Group")
            ax.set_ylabel("Average Life expectancy (years)")
            show_figure(fig)
        except Exception as exc:
            st.warning(f"Could not create GDP groups: {exc}")


# ------------------------------------------------------------
# 9. DEVELOPED VS DEVELOPING
# ------------------------------------------------------------

elif section == "Developed vs Developing":
    st.header("Life Expectancy by Country Status")
    st.caption("Comparison of life-expectancy distributions by development status.")


    if "Status" not in eda_df.columns:
        st.warning("Status column is not available.")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=eda_df,
            x="Status",
            y=TARGET,
            ax=ax
        )
        ax.set_title("Life Expectancy by Country Status")
        ax.set_xlabel("Country Status")
        ax.set_ylabel("Life expectancy (years)")
        show_figure(fig)

        status_summary = (
            eda_df.groupby("Status")[TARGET]
            .agg(["count", "mean", "median", "std"])
            .round(2)
            .reset_index()
        )
        st.subheader("Summary by Status")
        st.dataframe(status_summary, hide_index=True, use_container_width=True)

        if {"Developed", "Developing"}.issubset(set(status_summary["Status"].astype(str))):
            means = status_summary.set_index("Status")["mean"]
            gap = means["Developed"] - means["Developing"]
            st.info(
                f"**Result from this dataset:** average life expectancy is "
                f"{means['Developed']:.2f} years for Developed observations and "
                f"{means['Developing']:.2f} years for Developing observations "
                f"(difference: {gap:.2f} years). This describes the observed "
                f"groups and does not establish causation."
            )


# ------------------------------------------------------------
# 10. REGRESSION ANALYSIS — MATCHES CURRENT data.py
# ------------------------------------------------------------

elif section == "Regression Analysis":
    st.header("Schooling vs Life Expectancy")
    st.caption("Linear regression using Schooling to estimate life expectancy.")


    if "Schooling" not in df.columns or "Country" not in df.columns:
        st.warning("Required columns are not available.")
    else:
        regression_data = df[["Schooling", TARGET, "Country"]].dropna().copy()
        X_reg = regression_data[["Schooling"]]
        y_reg = regression_data[TARGET]
        groups_reg = regression_data["Country"]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=42
        )
        train_idx, test_idx = next(
            splitter.split(X_reg, y_reg, groups=groups_reg)
        )

        X_train = X_reg.iloc[train_idx]
        X_test = X_reg.iloc[test_idx]
        y_train = y_reg.iloc[train_idx]
        y_test = y_reg.iloc[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        r2 = r2_score(y_test, prediction)
        rmse = np.sqrt(mean_squared_error(y_test, prediction))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slope", f"{model.coef_[0]:.4f}")
        c2.metric("Intercept", f"{model.intercept_:.4f}")
        c3.metric("Test R²", f"{r2:.4f}")
        c4.metric("Test RMSE", f"{rmse:.4f}")

        st.write(
            f"Training rows: **{len(X_train):,}**  |  "
            f"Testing rows: **{len(X_test):,}**"
        )
        st.write(
            "Countries appearing in both sets: **"
            f"{len(set(groups_reg.iloc[train_idx]).intersection(set(groups_reg.iloc[test_idx])))}**"
        )

        order = np.argsort(X_test["Schooling"].values)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            X_test["Schooling"],
            y_test,
            alpha=0.4,
            label="Actual test observations"
        )
        ax.plot(
            X_test["Schooling"].values[order],
            prediction[order],
            color="red",
            label="Regression predictions"
        )
        ax.set_title("Schooling vs Life Expectancy — Test Set")
        ax.set_xlabel("Schooling (years)")
        ax.set_ylabel("Life expectancy (years)")
        ax.legend()
        show_figure(fig)

        st.caption(
            "The regression results are recalculated in this dashboard "
            "using the same country-grouped 80:20 split as data.py."
        )


# ------------------------------------------------------------
# 11. CLASSIFICATION OVERVIEW
# ------------------------------------------------------------

elif section == "Classification Overview":
    st.header("Machine-Learning Classification")
    st.caption("Classification of life expectancy into three project-defined categories.")
    class_counts = pd.cut(
        df[TARGET], bins=[-np.inf, 60, 75, np.inf],
        labels=["Low", "Medium", "High"], right=False
    ).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    st.write("**Observed class distribution in the cleaned dataset:**")
    st.dataframe(
        pd.DataFrame({
            "Class": class_counts.index.astype(str),
            "Observations": class_counts.values,
            "Share (%)": (class_counts.values / class_counts.sum() * 100).round(2)
        }),
        hide_index=True,
        use_container_width=True
    )


    st.subheader("Target Classes")
    class_df = pd.DataFrame({
        "Class": CLASS_LABELS,
        "Definition": [
            "Life expectancy < 60",
            "60 ≤ Life expectancy < 75",
            "Life expectancy ≥ 75"
        ]
    })
    st.dataframe(class_df, hide_index=True, use_container_width=True)

    st.subheader("Input Features")
    excluded = [TARGET, "Life Expectancy Class", "Country"]
    feature_columns = [
        col for col in df.columns if col not in excluded
    ]

    st.metric("Original input attributes", len(feature_columns))
    st.write(feature_columns)

    st.markdown(
        """
        **Country** is excluded from the model input features and retained
        separately for group-based splitting. The original numerical
        `Life expectancy` column is removed from the inputs because it is
        used to create the classification target.
        """
    )

    st.subheader("Algorithms")
    algo_table = pd.DataFrame({
        "Algorithm": [
            "Decision Tree",
            "K-Nearest Neighbors (KNN)",
            "Gaussian Naive Bayes",
            "Random Forest"
        ],
        "Main idea": [
            "Uses feature-based decision rules.",
            "Predicts from the majority class among the nearest 5 observations.",
            "Uses Bayes' theorem with a Gaussian feature assumption.",
            "Combines 200 decision trees using ensemble learning."
        ]
    })
    st.dataframe(algo_table, hide_index=True, use_container_width=True)

    st.subheader("Train-Test Experiments")
    st.dataframe(
        pd.DataFrame({
            "Split": ["80:20", "70:30", "75:25"],
            "Test size parameter": [0.20, 0.30, 0.25]
        }),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "GroupShuffleSplit selects groups (countries), so the actual "
        "percentage of rows in the test set may differ from the requested "
        "test_size percentage."
    )


# ------------------------------------------------------------
# 12. MODEL COMPARISON
# ------------------------------------------------------------

elif section == "Model Comparison":
    st.header("Final Model Comparison")
    st.caption("Comparison of four classifiers across three country-grouped splits.")


    if results_df.empty:
        st.warning(
            "Model results were not found. Run data.py first to generate "
            "`ml_experiment_results.csv`, then refresh this dashboard."
        )
    else:
        required = {"Model", "Split", "Accuracy", "Macro Precision",
                    "Macro Recall", "Macro F1"}
        if not required.issubset(results_df.columns):
            st.error(
                "The experiment results CSV does not contain the expected "
                "columns. Regenerate it by running the current data.py."
            )
        else:
            st.subheader("Results Across All Train-Test Splits")
            st.dataframe(
                results_df.round(4),
                hide_index=True,
                use_container_width=True
            )

            st.subheader("Macro F1 Comparison by Split")
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.barplot(
                data=results_df,
                x="Model",
                y="Macro F1",
                hue="Split",
                errorbar=None,
                ax=ax
            )
            ax.set_title("Macro F1 Comparison — Models and Splits")
            ax.set_ylabel("Macro F1-score")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Train-Test Split")
            plt.tight_layout()
            show_figure(fig)

            st.subheader("Average Model Performance")
            metrics = [
                "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
            ]
            model_summary = (
                results_df.groupby("Model")[metrics]
                .mean()
                .sort_values("Macro F1", ascending=False)
            )
            st.dataframe(
                model_summary.round(4),
                use_container_width=True
            )

            winner = model_summary.index[0]
            best_f1 = model_summary.iloc[0]["Macro F1"]
            best_accuracy = model_summary.loc[winner, "Accuracy"]

            st.success(
                f"Best model by average Macro F1: **{winner}** "
                f"(Macro F1 = {best_f1:.4f})"
            )
            st.metric("Winner's average accuracy", f"{best_accuracy:.4f}")

            st.caption(
                "The winner is selected using the average Macro F1 across "
                "the three country-grouped experiments. Check the separate "
                "future-year validation results as well."
            )


# ------------------------------------------------------------
# 13. CROSS-VALIDATION RESULTS
# ------------------------------------------------------------

elif section == "Cross-Validation Results":
    st.header("Group-Aware Five-Fold Cross-Validation")
    st.caption("Model consistency across country-grouped validation folds.")


    if cv_df.empty:
        st.warning(
            "Cross-validation results were not found. Run data.py first "
            "to generate `ml_cross_validation_results.csv`."
        )
    else:
        st.dataframe(
            cv_df.round(4),
            hide_index=True,
            use_container_width=True
        )

        if {"Model", "Mean CV Macro F1"}.issubset(cv_df.columns) and not cv_df.empty:
            best_cv = cv_df.groupby("Model")["Mean CV Macro F1"].mean().sort_values(ascending=False)
            st.info(
                f"**Result from these folds:** {best_cv.index[0]} has the highest "
                f"average cross-validation Macro F1 ({best_cv.iloc[0]:.4f}) "
                f"among the models listed. Compare the fold-level values as well "
                f"to judge consistency."
            )

        if {"Model", "Split", "Mean CV Macro F1"}.issubset(cv_df.columns):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=cv_df,
                x="Model",
                y="Mean CV Macro F1",
                hue="Split",
                errorbar=None,
                ax=ax
            )
            ax.set_title("Cross-Validation Mean Macro F1")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            show_figure(fig)

        st.caption(
            "The current data.py uses StratifiedGroupKFold. The number of "
            "folds can be lower than five when the training data has too "
            "few observations in its smallest class or too few countries."
        )


# ------------------------------------------------------------
# 14. FUTURE-YEAR VALIDATION
# ------------------------------------------------------------

elif section == "Future-Year Validation":
    st.header("Future-Year Validation")
    st.caption("Evaluation on the last three available years.")


    if future_df.empty:
        st.warning(
            "Future-year results were not found. Run data.py first to "
            "generate `future_year_results.csv`."
        )
    else:
        st.dataframe(
            future_df.round(4),
            hide_index=True,
            use_container_width=True
        )

        if {"Model", "Macro F1"}.issubset(future_df.columns) and not future_df.empty:
            best_future = future_df.loc[future_df["Macro F1"].idxmax()]
            st.info(
                f"**Result on the held-out later years:** {best_future['Model']} "
                f"has the highest Macro F1 ({best_future['Macro F1']:.4f}) "
                f"among the models shown. This identifies the strongest result "
                f"for this particular time-based test."
            )

        if {"Model", "Macro F1"}.issubset(future_df.columns):
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(
                data=future_df,
                x="Model",
                y="Macro F1",
                errorbar=None,
                ax=ax
            )
            ax.set_title("Future-Year Validation — Macro F1")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            show_figure(fig)

        st.caption(
            "This is a time-based evaluation. It is separate from the "
            "country-grouped train-test experiments."
        )


# ------------------------------------------------------------
# 15. DECISION TREE VISUALIZATION
# ------------------------------------------------------------

elif section == "Decision Tree Visualization":
    st.header("Decision Tree Visualization")
    st.caption("First three levels of a separately trained decision tree.")


    if TREE_IMAGE.exists():
        st.image(
            str(TREE_IMAGE),
            caption="Decision Tree — first three levels",
            use_container_width=True
        )
        st.caption(
            "This visualization is generated by data.py using a separately "
            "trained Decision Tree on the country-grouped 80:20 training "
            "partition. It displays the first three levels for readability."
        )
    else:
        st.warning(
            "`decision_tree_visualization.png` was not found. Run data.py "
            "first; it saves this image in the project folder."
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")
st.caption(
    "Life Expectancy Analysis Dashboard | Built with Python, "
    "Pandas, Scikit-learn, Matplotlib, Seaborn and Streamlit."
)
