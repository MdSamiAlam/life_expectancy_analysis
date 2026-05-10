# =============================================================================
#  Life Expectancy Analysis Dashboard
#  Streamlit version of the original analysis script.
#  All calculations, styles, and visualisation code are identical
#  to the original — only the output layer is changed (st.* instead of print/show).
#
#  HOW TO RUN:
#    1. Place this file in the same folder as "Life Expectancy Data.csv"
#    2. pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
#    3. streamlit run app.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── Global style (exactly as in original) ────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


# =============================================================================
#  DATA LOADING & CLEANING  — cached so it only runs once
# =============================================================================

@st.cache_data
def load_and_clean():
    df = pd.read_csv("Life Expectancy Data.csv")
    df.columns = df.columns.str.strip()

    # LOW missing → mean
    low_cols = ["Life expectancy", "Adult Mortality", "BMI", "Polio", "Diphtheria"]
    for col in low_cols:
        df[col] = df[col].fillna(df[col].mean())

    # MEDIUM missing → group mean
    medium_cols = ["Alcohol", "Total expenditure", "Schooling",
                   "Income composition of resources"]
    for col in medium_cols:
        df[col] = df.groupby("Status")[col].transform(
            lambda x: x.fillna(x.mean())
        )

    # HIGH missing → drop
    df.drop(columns=["Population"], inplace=True)

    # Remaining important columns → mean
    df["GDP"]         = df["GDP"].fillna(df["GDP"].mean())
    df["Hepatitis B"] = df["Hepatitis B"].fillna(df["Hepatitis B"].mean())

    # Outlier capping via IQR
    Q1  = df["Life expectancy"].quantile(0.25)
    Q3  = df["Life expectancy"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["Life expectancy"] = np.where(
        df["Life expectancy"] < lower_bound, lower_bound, df["Life expectancy"]
    )
    df["Life expectancy"] = np.where(
        df["Life expectancy"] > upper_bound, upper_bound, df["Life expectancy"]
    )

    # GDP groups (used in bar plot section)
    df["GDP_Group"] = pd.qcut(
        df["GDP"], q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    return df, lower_bound, upper_bound


# =============================================================================
#  PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Life Expectancy Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 Life Expectancy Analysis Dashboard")
st.markdown(
    "**Dataset:** WHO Life Expectancy (2000–2015) &nbsp;|&nbsp; "
    "**Source:** [Kaggle – kumarajarshi]"
    "(https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) "
    "&nbsp;|&nbsp; **Rows:** 2,938 &nbsp;|&nbsp; **Columns:** 22"
)
st.markdown("---")

# ── Load data ────────────────────────────────────────────────────────────────
try:
    df, lower_bound, upper_bound = load_and_clean()
except FileNotFoundError:
    st.error(
        "**`Life Expectancy Data.csv` not found.**  \n"
        "Place the CSV in the **same folder** as `app.py` and restart."
    )
    st.stop()

# =============================================================================
#  SIDEBAR
# =============================================================================

st.sidebar.title("Navigation")

SECTIONS = [
    "Project Objectives",
    "Data Preview",
    "Dataset Info",
    "Summary Statistics",
    "Missing Values",
    "Distribution Analysis",
    "Outlier Handling",
    "Correlation Matrix",
    "GDP vs Life Expectancy",
    "Developed vs Developing",
    "Correlation Heatmap",
    "Regression Analysis",
    "Hypothesis Testing",
]

section = st.sidebar.radio("Go to section:", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**After cleaning**  \nShape: `{df.shape[0]} rows × {df.shape[1]} cols`"
)
st.sidebar.markdown(
    f"**IQR Bounds**  \nLower: `{lower_bound:.1f}` &nbsp; Upper: `{upper_bound:.1f}`"
)


# =============================================================================
#  SECTIONS
# =============================================================================

# ─── 1. Project Objectives ────────────────────────────────────────────────────
if section == "Project Objectives":
    st.header("Project Objectives")

    objectives = """
  Objective 1 (Linear Regression):
    Analyse the impact of Schooling (years of education) on
    Life Expectancy across countries using simple linear regression.
    X = Schooling   |   Y = Life expectancy

  Objective 2 (Linear Regression):
    Examine the impact of Adult Mortality on Life Expectancy
    using simple linear regression.
    X = Adult Mortality   |   Y = Life expectancy

  Objective 3 (Visualisation - Bar Plot):
    Analyse how Life Expectancy varies across different GDP groups
    to understand the economic influence on health outcomes.

  Objective 4 (Visualisation - Box Plot):
    Compare the average Life Expectancy between
    Developed and Developing countries using a bar plot.

  Objective 5 (Visualisation - Heatmap):
    Identify which numerical health and economic indicators
    correlate most strongly with Life Expectancy using a
    correlation heatmap.
"""
    st.code(objectives, language=None)


# ─── 2. Data Preview ─────────────────────────────────────────────────────────
elif section == "Data Preview":
    st.header("Data Preview")
    st.markdown(f"**Dataset loaded:** `{df.shape}`")
    st.dataframe(df.head(), use_container_width=True)


# ─── 3. Dataset Info ─────────────────────────────────────────────────────────
elif section == "Dataset Info":
    st.header("Dataset Info")

    import io
    buf = io.StringIO()
    df.info(buf=buf)
    st.code(buf.getvalue(), language="text")


# ─── 4. Summary Statistics ───────────────────────────────────────────────────
elif section == "Summary Statistics":
    st.header("Summary Statistics (Numerical Columns)")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ─── 5. Missing Values ───────────────────────────────────────────────────────
elif section == "Missing Values":
    st.header("Missing Values")

    # Original raw file — reload without cleaning to show before state
    df_raw = pd.read_csv("Life Expectancy Data.csv")
    df_raw.columns = df_raw.columns.str.strip()

    st.subheader("Before Handling")
    missing_before = df_raw.isnull().sum().reset_index()
    missing_before.columns = ["Column", "Missing Count"]
    st.dataframe(missing_before, use_container_width=True)

    st.subheader("After Handling")
    missing_after = df.isnull().sum().reset_index()
    missing_after.columns = ["Column", "Missing Count"]
    st.dataframe(missing_after, use_container_width=True)

    st.markdown("""
    **Strategy used:**
    - **LOW missing** → filled with column mean *(Life expectancy, Adult Mortality, BMI, Polio, Diphtheria)*
    - **MEDIUM missing** → filled with group mean by Status *(Alcohol, Total expenditure, Schooling, Income composition)*
    - **HIGH missing** → column dropped *(Population)*
    - **Remaining** → filled with column mean *(GDP, Hepatitis B)*
    """)


# ─── 6. Distribution Analysis ────────────────────────────────────────────────
elif section == "Distribution Analysis":
    st.header("Distribution Analysis — Life Expectancy")

    st.markdown(
        "Histogram and Boxplot to check the distribution shape "
        "and identify skewness **before outlier handling**."
    )

    # 🔥 Load RAW data (before cleaning)
    df_raw = pd.read_csv("Life Expectancy Data.csv")
    df_raw.columns = df_raw.columns.str.strip()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df_raw["Life expectancy"], kde=True, ax=ax)
        ax.set_title("Histogram of Life Expectancy (Before Outliers)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=df_raw["Life expectancy"], ax=ax)
        ax.set_title("Boxplot of Life Expectancy (Before Outliers)")
        st.pyplot(fig)
        plt.close(fig)

    st.success(
        "✔ Outliers are clearly visible in the boxplot (points outside whiskers)."
    )

    st.info(
        "Data is negatively skewed and contains extreme values → "
        "outlier handling (IQR capping) is required."
    )

# ─── 7. Outlier Handling ─────────────────────────────────────────────────────
elif section == "Outlier Handling":
    st.header("Outlier Handling using IQR Method")

    st.markdown("**Capping** is used instead of removing outliers.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Lower Bound", f"{lower_bound:.4f}")
    col2.metric("Upper Bound", f"{upper_bound:.4f}")
    col3.metric("Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")

    st.subheader("Boxplot After Outlier Capping")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Life expectancy"], ax=ax)
    ax.set_title("Boxplot of Life Expectancy")
    st.pyplot(fig)
    plt.close(fig)


# ─── 8. Correlation Matrix ───────────────────────────────────────────────────
elif section == "Correlation Matrix":
    st.header("Correlation with Life Expectancy")

    num_df = df.select_dtypes(include=np.number)
    corr_matrix = num_df.corr()

    st.subheader("Full Correlation Matrix")
    st.dataframe(corr_matrix.round(3), use_container_width=True)

    st.subheader("Correlation with Life Expectancy (sorted)")
    corr_with_le = corr_matrix["Life expectancy"].drop("Life expectancy")
    corr_sorted = corr_with_le.sort_values(ascending=False).round(3)

    corr_df = corr_sorted.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    st.dataframe(corr_df, use_container_width=True)


# ─── 9. GDP vs Life Expectancy ────────────────────────────────────────────────
elif section == "GDP vs Life Expectancy":
    st.header("Bar Plot — GDP Groups vs Average Life Expectancy")
    st.markdown(
        "Objective 3: Analyse how Life Expectancy varies across different "
        "GDP groups to understand the economic influence on health outcomes."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="GDP_Group",
        y="Life expectancy",
        data=df,
        width=0.5,
        errorbar=None,
        ax=ax
    )
    ax.set_title("Average Life Expectancy by GDP Groups")
    ax.set_xlabel("GDP Groups")
    ax.set_ylabel("Average Life Expectancy")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("➡ Life expectancy increases with GDP groups")
    st.markdown("➡ Relationship is positive but not very strong")


# ─── 10. Developed vs Developing ─────────────────────────────────────────────
elif section == "Developed vs Developing":
    st.header("Box Plot — Life Expectancy by Country Status")
    st.markdown(
        "Objective 4: Compare the Life Expectancy distribution between "
        "Developed and Developing countries."
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df,
        x="Status",
        y="Life expectancy",
        palette=["#e74c3c", "#2ecc71"],
        width=0.5,
        ax=ax
    )
    ax.set_title("Life Expectancy Distribution by Country Status")
    ax.set_xlabel("Country Status")
    ax.set_ylabel("Life Expectancy")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("➡ Developed countries show higher median life expectancy")
    st.markdown("➡ Developing countries show more variability and lower values")


# ─── 11. Correlation Heatmap ─────────────────────────────────────────────────
elif section == "Correlation Heatmap":
    st.header("Correlation Heatmap — All Numerical Features")
    st.markdown(
        "Objective 5: Identify which numerical health and economic indicators "
        "correlate most strongly with Life Expectancy."
    )

    num_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        num_df.corr(),
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.5,
        cbar=True,
        ax=ax
    )
    ax.set_title("Correlation Heatmap (All Numerical Features)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─── 12. Regression Analysis ─────────────────────────────────────────────────
elif section == "Regression Analysis":
    st.header("Simple Linear Regression")

    # ── Exact regression_model() function from original code ─────────────────
    def regression_model(x_col, display_name=None):
        display_name = display_name or x_col

        data = df[[x_col, "Life expectancy"]].dropna()
        X = data[[x_col]].values
        y = data["Life expectancy"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2   = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Print block — shown as a code block in Streamlit
        result_text = (
            f"\n{display_name} → Life Expectancy\n"
            f"{'-'*40}\n"
            f"Slope:     {round(model.coef_[0], 4)}\n"
            f"Intercept: {round(model.intercept_, 4)}\n"
            f"R²:        {round(r2, 4)}\n"
            f"RMSE:      {round(rmse, 4)}"
        )
        st.code(result_text, language="text")

        # Plot — identical to original
        x_sorted = np.sort(X, axis=0)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(X, y, alpha=0.3)
        ax.plot(x_sorted, model.predict(x_sorted), color="red")
        ax.set_title(f"{display_name} vs Life Expectancy")
        ax.set_xlabel(display_name)
        ax.set_ylabel("Life Expectancy")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        return model.coef_[0]

    # Model 1 — Schooling
    st.subheader("Model 1 — Schooling → Life Expectancy")
    st.caption("Objective 1: Analyse the impact of Schooling on Life Expectancy.")
    s1 = regression_model("Schooling", "Schooling")
    st.markdown(
        f"➡ 1 year increase in schooling → **{s1:.2f} years** increase in life expectancy"
    )

    st.markdown("---")

    # Model 2 — Adult Mortality
    st.subheader("Model 2 — Adult Mortality → Life Expectancy")
    st.caption("Objective 2: Examine the impact of Adult Mortality on Life Expectancy.")
    s2 = regression_model("Adult Mortality", "Adult Mortality")
    st.markdown(
        f"➡ Increase in adult mortality → **{s2:.2f}** decrease in life expectancy"
    )


# ─── 13. Hypothesis Testing ───────────────────────────────────────────────────
elif section == "Hypothesis Testing":
    st.header("Hypothesis Testing")
    st.markdown("""
    **H₀:** No significant difference in Life Expectancy between Developed and Developing countries.  
    **H₁:** Developed countries have significantly higher Life Expectancy.  
    **Test:** Welch's Independent T-Test &nbsp;|&nbsp; **α = 0.05** &nbsp;|&nbsp; One-tailed
    """)

    developed  = df[df["Status"] == "Developed"]["Life expectancy"]
    developing = df[df["Status"] == "Developing"]["Life expectancy"]

    t_stat, p_val = stats.ttest_ind(
        developed, developing,
        equal_var=False,
        alternative="greater"
    )

    # Results — same as original print block
    result_text = (
        f"Mean Developed:  {round(developed.mean(), 2)}\n"
        f"Mean Developing: {round(developing.mean(), 2)}\n"
        f"T-stat:          {round(t_stat, 4)}\n"
        f"P-value:         {p_val}"
    )
    st.code(result_text, language="text")

    if p_val < 0.05:
        st.success("Reject H₀ → Developed countries have higher life expectancy")
    else:
        st.warning("Fail to reject H₀")

    st.markdown("---")

    # Bar chart — identical to original
    st.subheader("Life Expectancy Comparison — Bar Chart")

    means  = [developed.mean(), developing.mean()]
    labels = ["Developed", "Developing"]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, means, width=0.4)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center"
        )

    ax.set_title("Life Expectancy Comparison")
    ax.set_ylabel("Life Expectancy")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)