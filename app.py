from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "Groceries_dataset.csv"
RULES_PATH = BASE_DIR / "association_rules.csv"
MATRIX_PATH = BASE_DIR / "transaction_matrix.csv"


st.set_page_config(
    page_title="Market Basket Analysis Dashboard",
    page_icon="🧺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
#  Custom CSS for a professional SaaS look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #176d60;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section headers */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #182021;
        margin-top: 24px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #176d60;
        display: inline-block;
    }

    /* Info / help boxes */
    .help-box {
        background-color: #f0f7f6;
        border-left: 4px solid #176d60;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
    }

    /* Chart container */
    .chart-box {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }

    /* Streamlit tabs override for cleaner look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
#  Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    data = pd.read_csv(RAW_DATA_PATH)
    data["itemDescription"] = data["itemDescription"].astype(str).str.strip().str.lower()
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="coerce")
    data["transaction_id"] = data["Member_number"].astype(str) + "_" + data["Date"].dt.strftime("%Y-%m-%d")
    data["month"] = data["Date"].dt.to_period("M").astype(str)
    data["day_name"] = data["Date"].dt.day_name()
    return data


@st.cache_data(show_spinner=False)
def load_rules() -> pd.DataFrame:
    rules = pd.read_csv(RULES_PATH)
    numeric_columns = ["Support", "Confidence", "Lift", "Leverage", "Conviction", "Jaccard"]
    for column in numeric_columns:
        rules[column] = pd.to_numeric(rules[column], errors="coerce")
    rules["Antecedent"] = rules["Antecedent"].astype(str).str.strip().str.lower()
    rules["Consequent"] = rules["Consequent"].astype(str).str.strip().str.lower()
    rules["Rule"] = rules["Antecedent"] + " -> " + rules["Consequent"]
    return rules.sort_values(["Lift", "Confidence", "Support"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_matrix_columns() -> list[str]:
    header = pd.read_csv(MATRIX_PATH, nrows=0)
    return [column for column in header.columns if column != "transaction_id"]


@st.cache_data(show_spinner=False)
def build_transaction_items(data: pd.DataFrame) -> pd.Series:
    return data.groupby("transaction_id")["itemDescription"].apply(lambda values: sorted(set(values)))


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def basket_segments(transaction_items: pd.Series) -> pd.DataFrame:
    sizes = transaction_items.apply(len)
    labels = pd.cut(
        sizes,
        bins=[0, 1, 3, 6, np.inf],
        labels=["single-item", "small basket", "medium basket", "large basket"],
        right=True,
    )
    segment = (
        pd.DataFrame({"segment": labels, "basket_size": sizes})
        .groupby("segment", observed=False)
        .agg(transactions=("basket_size", "count"), average_size=("basket_size", "mean"))
        .reset_index()
    )
    segment["share"] = segment["transactions"] / segment["transactions"].sum()
    return segment


def recommend_from_rules(selected_items: list[str], rules: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = {item.lower() for item in selected_items}
    rows = []

    for _, rule in rules.iterrows():
        antecedents = {item.strip() for item in str(rule["Antecedent"]).split(",") if item.strip()}
        consequents = {item.strip() for item in str(rule["Consequent"]).split(",") if item.strip()}
        if antecedents and antecedents.issubset(selected):
            for item in consequents - selected:
                rows.append(
                    {
                        "Recommended product": item,
                        "Triggered by": ", ".join(sorted(antecedents)),
                        "Support": rule["Support"],
                        "Confidence": rule["Confidence"],
                        "Lift": rule["Lift"],
                        "Reason": (
                            f"Customers buying {', '.join(sorted(antecedents))} were "
                            f"{rule['Lift']:.2f}x more likely to also buy {item}."
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame()

    recommendations = pd.DataFrame(rows)
    recommendations = (
        recommendations.sort_values(["Lift", "Confidence", "Support"], ascending=False)
        .drop_duplicates(subset=["Recommended product"], keep="first")
        .head(top_n)
        .reset_index(drop=True)
    )
    return recommendations


def fallback_cooccurrence(selected_items: list[str], transaction_items: pd.Series, top_n: int) -> pd.DataFrame:
    selected = {item.lower() for item in selected_items}
    counts: dict[str, int] = {}
    matching_transactions = 0

    for items in transaction_items:
        basket = set(items)
        if selected.issubset(basket):
            matching_transactions += 1
            for item in basket - selected:
                counts[item] = counts.get(item, 0) + 1

    if not counts:
        return pd.DataFrame()

    total_transactions = len(transaction_items)
    rows = [
        {
            "Recommended product": item,
            "Co-occurrence count": count,
            "Basket match rate": count / matching_transactions if matching_transactions else 0,
            "Dataset support": count / total_transactions,
        }
        for item, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(["Co-occurrence count", "Dataset support"], ascending=False).head(top_n)


def to_excel(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
#  Sidebar – Platform Selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Platform Selector")
    st.markdown("Choose the analytics experience you want.")
    dashboard_mode = st.radio(
        label="",
        options=[
            "🚀 Market Basket Engine",
            "📊 Power BI Preview",
            "❄️ Snowflake Cloud",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:12px; color:#666;">
        <b>How it works:</b><br>
        • <b>Market Basket Engine</b> — Interactive rules & recommendations.<br>
        • <b>Power BI Preview</b> — Live dashboard preview + export tools.<br>
        • <b>Snowflake Cloud</b> — Cloud warehouse setup guide.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
#  Load data once (used by every mode)
# ---------------------------------------------------------------------------
raw_data = load_raw_data()
rules = load_rules()
matrix_items = load_matrix_columns()
transaction_items = build_transaction_items(raw_data)


# =============================================================================
#  MODE 1 – Market Basket Engine (Streamlit)
# =============================================================================
if dashboard_mode == "🚀 Market Basket Engine":
    st.markdown("<h1 style='margin-bottom:4px;'>Market Basket Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:24px;'>Groceries transaction analysis, association rules, and product recommendations.</p>", unsafe_allow_html=True)

    tab_overview, tab_rules, tab_recommender, tab_segments, tab_deploy = st.tabs(
        ["📈 Overview", "🔗 Rules", "🎯 Recommendations", "🧺 Segments", "🚀 Deploy"]
    )

    with tab_overview:
        # Metric cards row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Purchase Records", f"{len(raw_data):,}")
        with c2:
            metric_card("Unique Transactions", f"{raw_data['transaction_id'].nunique():,}")
        with c3:
            metric_card("Unique Products", f"{raw_data['itemDescription'].nunique():,}")
        with c4:
            metric_card("Association Rules", f"{len(rules):,}")

        st.markdown("<div class='section-title'>Top Products</div>", unsafe_allow_html=True)
        top_n = st.slider("Number of products to show", min_value=5, max_value=30, value=15, key="top_products")
        top_items = raw_data["itemDescription"].value_counts().head(top_n)
        st.bar_chart(top_items, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-title'>Monthly Volume</div>", unsafe_allow_html=True)
            monthly = raw_data.groupby("month").size()
            st.line_chart(monthly, use_container_width=True)
        with col_b:
            st.markdown("<div class='section-title'>Transaction Size Distribution</div>", unsafe_allow_html=True)
            size_counts = transaction_items.apply(len).value_counts().sort_index()
            st.bar_chart(size_counts, use_container_width=True)

    with tab_rules:
        st.markdown("<div class='section-title'>Association Rules Explorer</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            min_lift = st.slider("Minimum Lift", 1.0, float(max(1.0, rules["Lift"].max())), 1.0, 0.01)
        with c2:
            min_confidence = st.slider("Minimum Confidence", 0.0, float(max(0.01, rules["Confidence"].max())), 0.0, 0.01)

        filtered_rules = rules[(rules["Lift"] >= min_lift) & (rules["Confidence"] >= min_confidence)].copy()

        st.dataframe(
            filtered_rules[
                ["Rank", "Rule", "Support", "Confidence", "Lift", "Leverage", "Conviction", "Jaccard"]
            ].style.format(
                {
                    "Support": "{:.4f}",
                    "Confidence": "{:.4f}",
                    "Lift": "{:.4f}",
                    "Leverage": "{:.5f}",
                    "Conviction": "{:.4f}",
                    "Jaccard": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-title'>Rules by Lift</div>", unsafe_allow_html=True)
            lift_chart = filtered_rules.set_index("Rule")["Lift"] if not filtered_rules.empty else pd.Series(dtype=float)
            st.bar_chart(lift_chart, use_container_width=True)
        with col_b:
            st.markdown("<div class='section-title'>Metric Summary</div>", unsafe_allow_html=True)
            if filtered_rules.empty:
                st.info("No rules match the selected filters.")
            else:
                st.dataframe(
                    filtered_rules[["Support", "Confidence", "Lift", "Leverage", "Conviction", "Jaccard"]]
                    .describe()
                    .round(4),
                    use_container_width=True,
                )

        st.markdown(
            """
            <div class="help-box">
            <b>How to read these rules:</b> Lift above 1 means the antecedent and consequent appear together 
            more often than expected under independence. The current dataset has small, sparse baskets, 
            so confidence values are generally low.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_recommender:
        st.markdown("<div class='section-title'>Product Recommendation Engine</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            selected_items = st.multiselect(
                "Select products currently in the basket",
                options=sorted(matrix_items),
                default=["sausage"] if "sausage" in matrix_items else [],
                placeholder="Choose products...",
            )
        with c2:
            top_n_recs = st.slider("Max recommendations", min_value=1, max_value=10, value=5)

        if selected_items:
            recommendations = recommend_from_rules(selected_items, rules, top_n_recs)
            if recommendations.empty:
                st.warning("No rule-based recommendation found. Showing co-occurrence fallback.")
                fallback = fallback_cooccurrence(selected_items, transaction_items, top_n_recs)
                if fallback.empty:
                    st.info("No co-occurring products found for this basket.")
                else:
                    st.dataframe(
                        fallback.style.format(
                            {
                                "Basket match rate": "{:.2%}",
                                "Dataset support": "{:.2%}",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.success(f"Found {len(recommendations)} recommendation(s) based on your basket.")
                st.dataframe(
                    recommendations.style.format(
                        {
                            "Support": "{:.4f}",
                            "Confidence": "{:.4f}",
                            "Lift": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Select at least one product to generate recommendations.")

        st.markdown("<div class='section-title'>Ready-Made Examples</div>", unsafe_allow_html=True)
        example_rows = []
        for basket in [["frankfurter"], ["other vegetables"], ["yogurt"], ["sausage"], ["soda"]]:
            result = recommend_from_rules(basket, rules, 3)
            if not result.empty:
                example_rows.append(
                    {
                        "Input basket": ", ".join(basket),
                        "Top recommendation": result.iloc[0]["Recommended product"],
                        "Lift": result.iloc[0]["Lift"],
                        "Confidence": result.iloc[0]["Confidence"],
                    }
                )
        st.dataframe(
            pd.DataFrame(example_rows).style.format({"Lift": "{:.4f}", "Confidence": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with tab_segments:
        st.markdown("<div class='section-title'>Basket Size Segmentation</div>", unsafe_allow_html=True)
        segments = basket_segments(transaction_items)
        st.dataframe(
            segments.style.format({"average_size": "{:.2f}", "share": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(segments.set_index("segment")["transactions"], use_container_width=True)

        st.markdown(
            """
            <div class="help-box">
            <b>Insight:</b> Most baskets are small (1-3 items), which explains why association rules have 
            low confidence. Low confidence is a data characteristic here, not automatically a failed model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_deploy:
        st.markdown("<div class='section-title'>Deployment Checklist</div>", unsafe_allow_html=True)
        st.markdown(
            """
            1. Push all project files to GitHub (`app.py`, `requirements.txt`, `.streamlit/config.toml`, and the CSV files).
            2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in with GitHub.
            3. Click **New app** and select this repository.
            4. Set the main file path to `app.py`.
            5. Click **Deploy**.
            """
        )


# =============================================================================
#  MODE 2 – Power BI Preview
# =============================================================================
elif dashboard_mode == "📊 Power BI Preview":
    st.markdown("<h1 style='margin-bottom:4px;'>Power BI Dashboard Preview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:24px;'>Live preview of your basket analytics + export tools for Power BI Desktop.</p>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    #  EXPLANATION BOX: What is this page?
    # -----------------------------------------------------------------------
    with st.expander("❓ What is this page and how do I use it?", expanded=True):
        st.markdown(
            """
            This page does **two things**:

            **1. Live Preview Dashboard** (below) — See your data visualized with interactive charts 
            directly in the browser. This is a preview of what your Power BI report could look like.

            **2. Export & Connect** (further below) — Download the cleaned data as CSV/Excel so you can 
            import it into **Power BI Desktop** and build your own professional report.

            **What is an Embedded Power BI Report?**
            After you build a report in **Power BI Desktop**, you can **Publish** it to the cloud 
            (**Power BI Service** at `app.powerbi.com`). Once published, Power BI gives you a special 
            **Embed URL** that looks like:
            ```
            https://app.powerbi.com/reportEmbed?reportId=abc123...
            ```
            If you paste that URL in the box at the bottom of this page, your live Power BI report 
            will appear directly inside this app.
            """
        )

    # -----------------------------------------------------------------------
    #  LIVE PREVIEW DASHBOARD (This is the actual dashboard part!)
    # -----------------------------------------------------------------------
    st.markdown("<div class='section-title'>Live Preview Dashboard</div>", unsafe_allow_html=True)

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{len(raw_data):,}")
    with c2:
        metric_card("Transactions", f"{raw_data['transaction_id'].nunique():,}")
    with c3:
        metric_card("Products", f"{raw_data['itemDescription'].nunique():,}")
    with c4:
        metric_card("Avg Basket Size", f"{transaction_items.apply(len).mean():.1f}")

    # Top products + Monthly trend
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title'>Top 15 Products</div>", unsafe_allow_html=True)
        top_15 = raw_data["itemDescription"].value_counts().head(15)
        st.bar_chart(top_15, use_container_width=True)
    with col_b:
        st.markdown("<div class='section-title'>Monthly Transaction Volume</div>", unsafe_allow_html=True)
        monthly_tx = raw_data.groupby("month")["transaction_id"].nunique()
        st.line_chart(monthly_tx, use_container_width=True)

    # Rules scatter plot + Basket segments
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div class='section-title'>Rules: Confidence vs Lift</div>", unsafe_allow_html=True)
        if not rules.empty:
            scatter_data = rules[["Confidence", "Lift", "Support", "Rule"]].copy()
            scatter_data["Size"] = scatter_data["Support"] * 5000  # Scale for visibility
            st.scatter_chart(
                scatter_data,
                x="Confidence",
                y="Lift",
                size="Size",
                color="Support",
                use_container_width=True,
            )
    with col_d:
        st.markdown("<div class='section-title'>Basket Segments</div>", unsafe_allow_html=True)
        seg = basket_segments(transaction_items)
        st.bar_chart(seg.set_index("segment")["transactions"], use_container_width=True)

    # Rules table preview
    st.markdown("<div class='section-title'>Top Association Rules</div>", unsafe_allow_html=True)
    top_rules = rules.head(20)[["Rank", "Rule", "Support", "Confidence", "Lift"]]
    st.dataframe(
        top_rules.style.format({"Support": "{:.4f}", "Confidence": "{:.4f}", "Lift": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
        height=300,
    )

    st.divider()

    # -----------------------------------------------------------------------
    #  EXPORT DATA SECTION
    # -----------------------------------------------------------------------
    st.markdown("<div class='section-title'>Export Data for Power BI Desktop</div>", unsafe_allow_html=True)
    st.write("Download these files and import them into Power BI Desktop using **Get Data → Text/CSV**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="📥 Raw transactions (CSV)",
            data=raw_data.to_csv(index=False).encode("utf-8"),
            file_name="groceries_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            label="📥 Association rules (CSV)",
            data=rules.to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        matrix_df = pd.read_csv(MATRIX_PATH)
        st.download_button(
            label="📥 Transaction matrix (CSV)",
            data=matrix_df.to_csv(index=False).encode("utf-8"),
            file_name="transaction_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.download_button(
        label="📥 All rules + metrics (Excel)",
        data=to_excel(rules, "AssociationRules"),
        file_name="association_rules.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()

    # -----------------------------------------------------------------------
    #  DAX MEASURES
    # -----------------------------------------------------------------------
    st.markdown("<div class='section-title'>DAX Measures for Power BI</div>", unsafe_allow_html=True)
    st.write("Copy these formulas into Power BI Desktop under **Modeling → New Measure**.")
    dax_measures = """
Total Transactions = DISTINCTCOUNT('groceries_cleaned'[transaction_id])

Total Purchase Records = COUNTROWS('groceries_cleaned')

Unique Products = DISTINCTCOUNT('groceries_cleaned'[itemDescription])

Avg Basket Size =
AVERAGEX(
    VALUES('groceries_cleaned'[transaction_id]),
    CALCULATE(DISTINCTCOUNT('groceries_cleaned'[itemDescription]))
)

Avg Lift = AVERAGE('association_rules'[Lift])

Max Confidence = MAX('association_rules'[Confidence])

Rule Count = COUNTROWS('association_rules')
"""
    st.code(dax_measures, language="dax")

    st.divider()

    # -----------------------------------------------------------------------
    #  EMBED LIVE REPORT
    # -----------------------------------------------------------------------
    st.markdown("<div class='section-title'>Embed Live Power BI Report</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="help-box">
        <b>How to get your Embed URL:</b><br>
        1. Build your report in <b>Power BI Desktop</b>.<br>
        2. Click <b>Publish</b> and sign in to your Power BI account.<br>
        3. Go to <b>Power BI Service</b> (app.powerbi.com) and open your published report.<br>
        4. Click <b>File → Embed report → Website or portal</b>.<br>
        5. Copy the URL that looks like <code>https://app.powerbi.com/reportEmbed?reportId=...</code><br>
        6. Paste it below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    embed_url = st.text_input(
        "Power BI Embed URL",
        value="",
        placeholder="https://app.powerbi.com/reportEmbed?reportId=...",
    )
    if embed_url:
        st.components.v1.iframe(embed_url, height=600, scrolling=True)
    else:
        st.info("Paste a valid Power BI Service embed URL above to render your live report here.")


# =============================================================================
#  MODE 3 – Snowflake Data Cloud
# =============================================================================
elif dashboard_mode == "❄️ Snowflake Cloud":
    st.markdown("<h1 style='margin-bottom:4px;'>Snowflake Data Cloud Setup</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:24px;'>Warehouse schema, connection code, and SQL scripts for cloud analytics.</p>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="help-box">
        <b>Why Snowflake?</b> Snowflake gives you elastic compute, automatic scaling, and the ability 
        to run SQL analytics or connect Power BI / Tableau directly to your warehouse. 
        You can sign up for a <a href="https://signup.snowflake.com/" target="_blank">free 30-day trial</a> 
        with $400 in credits.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_sql, tab_python, tab_copy, tab_queries, tab_arch = st.tabs(
        ["🗄️ SQL Schema", "🐍 Python Loader", "📤 Bulk Load", "📊 Analytics Queries", "🏗️ Architecture"]
    )

    with tab_sql:
        st.markdown("<div class='section-title'>SQL DDL — Create Tables</div>", unsafe_allow_html=True)
        sql_ddl = """
-- Create database and schema
CREATE DATABASE IF NOT EXISTS market_basket_db;
CREATE SCHEMA IF NOT EXISTS market_basket_db.analytics;
USE SCHEMA market_basket_db.analytics;

-- Raw transactions
CREATE OR REPLACE TABLE groceries_raw (
    Member_number   INTEGER,
    Date            DATE,
    itemDescription VARCHAR(255),
    transaction_id  VARCHAR(50),
    month           VARCHAR(7),
    day_name        VARCHAR(20)
);

-- Association rules
CREATE OR REPLACE TABLE association_rules (
    Rank        INTEGER,
    Antecedent  VARCHAR(500),
    Consequent  VARCHAR(500),
    Support     FLOAT,
    Confidence  FLOAT,
    Lift        FLOAT,
    Leverage    FLOAT,
    Conviction  FLOAT,
    Jaccard     FLOAT
);

-- Transaction matrix (wide format)
CREATE OR REPLACE TABLE transaction_matrix (
    transaction_id VARCHAR(50)
    -- product columns will be added dynamically or stored as a VARIANT
);
"""
        st.code(sql_ddl, language="sql")

    with tab_python:
        st.markdown("<div class='section-title'>Python Connector Snippet</div>", unsafe_allow_html=True)
        python_snippet = """
import snowflake.connector
import pandas as pd

# Replace with your Snowflake credentials (use Streamlit secrets in production)
conn = snowflake.connector.connect(
    account="YOUR_ACCOUNT",
    user="YOUR_USER",
    password="YOUR_PASSWORD",
    warehouse="COMPUTE_WH",
    database="market_basket_db",
    schema="analytics",
    role="ACCOUNTADMIN",
)

# Write a DataFrame
raw_data = pd.read_csv("Groceries_dataset.csv")
# ... clean it exactly like the Streamlit loader ...

success, nchunks, nrows, _ = conn.write_pandas(
    raw_data, "groceries_raw", overwrite=True
)
print(f"Loaded {nrows} rows into Snowflake.")
"""
        st.code(python_snippet, language="python")
        st.info(
            "In production, never hard-code credentials. Use Streamlit secrets (`secrets.toml`) or "
            "environment variables and load them via `st.secrets['snowflake']`."
        )

    with tab_copy:
        st.markdown("<div class='section-title'>Bulk Load via Stage (CSV → Snowflake)</div>", unsafe_allow_html=True)
        copy_sql = """
-- Upload CSV files to a Snowflake Stage first (e.g. via Snowsight UI or PUT command)

COPY INTO groceries_raw
FROM @my_stage/groceries_cleaned.csv
FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1, FIELD_OPTIONALLY_ENCLOSED_BY = '"');

COPY INTO association_rules
FROM @my_stage/association_rules.csv
FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1, FIELD_OPTIONALLY_ENCLOSED_BY = '"');
"""
        st.code(copy_sql, language="sql")

    with tab_queries:
        st.markdown("<div class='section-title'>Sample Analytics Queries</div>", unsafe_allow_html=True)
        analytics_sql = """
-- Top products
SELECT itemDescription, COUNT(*) AS purchases
FROM groceries_raw
GROUP BY itemDescription
ORDER BY purchases DESC
LIMIT 20;

-- Monthly volume
SELECT month, COUNT(DISTINCT transaction_id) AS transactions
FROM groceries_raw
GROUP BY month
ORDER BY month;

-- Strongest rules
SELECT Antecedent, Consequent, Lift, Confidence, Support
FROM association_rules
WHERE Lift > 1.0
ORDER BY Lift DESC;

-- Basket size distribution
SELECT
    CASE
        WHEN basket_size = 1 THEN 'single-item'
        WHEN basket_size BETWEEN 2 AND 3 THEN 'small basket'
        WHEN basket_size BETWEEN 4 AND 6 THEN 'medium basket'
        ELSE 'large basket'
    END AS segment,
    COUNT(*) AS transactions,
    AVG(basket_size) AS avg_size
FROM (
    SELECT transaction_id, COUNT(DISTINCT itemDescription) AS basket_size
    FROM groceries_raw
    GROUP BY transaction_id
)
GROUP BY segment
ORDER BY transactions DESC;
"""
        st.code(analytics_sql, language="sql")

    with tab_arch:
        st.markdown("<div class='section-title'>Architecture Overview</div>", unsafe_allow_html=True)
        st.text(
            """
    +-----------------+        +------------------+        +------------------+
    |  CSV Data       |  PUT   |  Snowflake       |  SQL   |  Power BI /      |
    |  (Groceries)    | -----> |  Stage → Tables  | -----> |  Tableau /       |
    +-----------------+        +------------------+        |  Streamlit       |
                                                          +------------------+
    
    Benefits:
    - Elastic storage (no local .csv bloat)
    - Concurrent read/write without locking
    - Direct Power BI connector (Import or DirectQuery)
    - Scales to billions of rows automatically
            """
        )

    st.divider()
    st.success(
        "Once your data is in Snowflake, return to **Power BI Preview** mode to download CSVs "
        "or connect Power BI directly using the Snowflake connector."
    )
