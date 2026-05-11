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
    page_icon="basket",
    layout="wide",
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
def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


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


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def to_excel(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
#  Sidebar – Dashboard selector (SaaS feel)
# ---------------------------------------------------------------------------
st.sidebar.title("Dashboard Selector")
dashboard_mode = st.sidebar.radio(
    "Choose your analytics platform:",
    options=[
        "Market Basket Engine (Streamlit)",
        "Power BI Dashboard",
        "Snowflake Data Cloud",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Switch between the built-in Streamlit engine, a Power BI export workspace, "
    "or a Snowflake warehouse setup guide."
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
if dashboard_mode == "Market Basket Engine (Streamlit)":
    st.title("Market Basket Analysis Dashboard")
    st.caption("Groceries transaction analysis, association rules, and product recommendations.")

    tab_overview, tab_rules, tab_recommender, tab_segments, tab_deploy = st.tabs(
        ["Overview", "Rules", "Recommendation Engine", "Basket Segments", "Deployment"]
    )

    with tab_overview:
        left, middle, right, far_right = st.columns(4)
        with left:
            metric_card("Purchase records", f"{len(raw_data):,}")
        with middle:
            metric_card("Transactions", f"{raw_data['transaction_id'].nunique():,}")
        with right:
            metric_card("Unique products", f"{raw_data['itemDescription'].nunique():,}")
        with far_right:
            metric_card("Exported rules", f"{len(rules):,}")

        st.divider()
        st.subheader("Top products")
        top_n = st.slider("Number of products", min_value=5, max_value=30, value=15, key="top_products")
        top_items = raw_data["itemDescription"].value_counts().head(top_n)
        st.bar_chart(top_items)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Monthly purchase volume")
            monthly = raw_data.groupby("month").size()
            st.line_chart(monthly)
        with col_b:
            st.subheader("Transaction size distribution")
            size_counts = transaction_items.apply(len).value_counts().sort_index()
            st.bar_chart(size_counts)

    with tab_rules:
        st.subheader("Association rules")
        min_lift = st.slider("Minimum lift", 1.0, float(max(1.0, rules["Lift"].max())), 1.0, 0.01)
        min_confidence = st.slider("Minimum confidence", 0.0, float(max(0.01, rules["Confidence"].max())), 0.0, 0.01)
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
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Rules ranked by lift")
            lift_chart = filtered_rules.set_index("Rule")["Lift"] if not filtered_rules.empty else pd.Series(dtype=float)
            st.bar_chart(lift_chart)
        with col_b:
            st.subheader("Metric summary")
            if filtered_rules.empty:
                st.info("No rules match the selected filters.")
            else:
                st.dataframe(
                    filtered_rules[["Support", "Confidence", "Lift", "Leverage", "Conviction", "Jaccard"]]
                    .describe()
                    .round(4),
                    use_container_width=True,
                )

        st.subheader("Interpretation")
        st.write(
            "Lift above 1 means the antecedent and consequent appear together more often than expected "
            "under independence. The current dataset has small, sparse baskets, so confidence values are low."
        )

    with tab_recommender:
        st.subheader("Product recommendation engine")
        selected_items = st.multiselect(
            "Basket products",
            options=sorted(matrix_items),
            default=["sausage"] if "sausage" in matrix_items else [],
        )
        top_n_recs = st.slider("Maximum recommendations", min_value=1, max_value=10, value=5)

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

        st.subheader("Ready examples")
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
        st.subheader("Basket size segmentation")
        segments = basket_segments(transaction_items)
        st.dataframe(
            segments.style.format({"average_size": "{:.2f}", "share": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(segments.set_index("segment")["transactions"])

        st.subheader("Segment interpretation")
        st.write(
            "Most baskets are small, which explains why association rules have low confidence. "
            "Low confidence is a data characteristic here, not automatically a failed model."
        )

    with tab_deploy:
        st.subheader("Deployment checklist")
        st.markdown(
            """
1. Push `app.py`, `requirements.txt`, `.streamlit/config.toml`, `association_rules.csv`, `transaction_matrix.csv`, and `Groceries_dataset.csv`.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.
"""
        )

        st.subheader("Project ownership note")
        st.write(
            "This dashboard covers the engine and deployment milestone by turning the mined rules into "
            "an interactive product recommendation interface with rule exploration and basket segmentation."
        )


# =============================================================================
#  MODE 2 – Power BI Dashboard
# =============================================================================
elif dashboard_mode == "Power BI Dashboard":
    st.title("Power BI Dashboard Export & Guide")
    st.caption("Export your data and DAX formulas to build a Power BI basket-analysis report.")

    st.markdown(
        """
        **Reference playlist:**
        [Basket Analysis Introduction – Best Practice Tips For Power BI Using DAX]
        (https://www.youtube.com/watch?v=z9ttZAZkEhs&list=PLM9ZnQfU_UGWxn7D9VRkKcMtcYt1wTCBf)
        """
    )

    st.divider()

    # -----------------------------------------------------------------------
    #  Export data
    # -----------------------------------------------------------------------
    st.subheader("1. Export cleaned data")
    st.write("Download the datasets below and import them into Power BI using **Get Data → Text/CSV**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="Raw transactions (CSV)",
            data=raw_data.to_csv(index=False).encode("utf-8"),
            file_name="groceries_cleaned.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            label="Association rules (CSV)",
            data=rules.to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            label="Transaction matrix (CSV)",
            data=pd.read_csv(MATRIX_PATH).to_csv(index=False).encode("utf-8"),
            file_name="transaction_matrix.csv",
            mime="text/csv",
        )

    st.download_button(
        label="All rules + metrics (Excel)",
        data=to_excel(rules, "AssociationRules"),
        file_name="association_rules.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()

    # -----------------------------------------------------------------------
    #  Recommended Power BI visuals
    # -----------------------------------------------------------------------
    st.subheader("2. Recommended Power BI visuals")
    st.markdown(
        """
| Visual | Fields | Insight |
|---|---|---|
| **Matrix / Table** | Antecedent → Consequent, Lift, Confidence, Support | Browse rules like the Streamlit table |
| **Scatter plot** | Confidence (X), Lift (Y), Support (Size) | Identify strong rules at a glance |
| **Bar chart** | Top items by purchase count | Same as *Top products* in Streamlit |
| **Line chart** | Month → Count of transaction_id | Monthly volume trend |
| **Funnel / Donut** | Basket-size segment → % of transactions | Segmentation view |
| **Slicer** | Antecedent | Dynamic filter to show only rules triggered by a selected product |
"""
    )

    st.divider()

    # -----------------------------------------------------------------------
    #  DAX measures
    # -----------------------------------------------------------------------
    st.subheader("3. DAX measures (copy-paste into Power BI)")
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
    #  Embed placeholder
    # -----------------------------------------------------------------------
    st.subheader("4. Embedded Power BI report (optional)")
    st.write(
        "If you publish your report to Power BI Service, paste the embed URL below to display it live."
    )
    embed_url = st.text_input("Power BI embed URL", value="", placeholder="https://app.powerbi.com/reportEmbed?reportId=...")
    if embed_url:
        st.components.v1.iframe(embed_url, height=600, scrolling=True)
    else:
        st.info("Enter a valid Power BI Service embed URL above to render the report inline.")


# =============================================================================
#  MODE 3 – Snowflake Data Cloud
# =============================================================================
elif dashboard_mode == "Snowflake Data Cloud":
    st.title("Snowflake Data Cloud Setup")
    st.caption("Warehouse schema, connection snippets, and SQL scripts for cloud analytics.")

    st.markdown(
        """
        **Why Snowflake?**
        Snowflake gives you elastic compute, automatic scaling, and the ability to run SQL analytics
        or connect Power BI / Tableau directly to your warehouse. You can sign up for a
        [free 30-day trial](https://signup.snowflake.com/) with $400 in credits.
        """
    )

    st.divider()

    # -----------------------------------------------------------------------
    #  SQL DDL
    # -----------------------------------------------------------------------
    st.subheader("1. SQL DDL – create tables")
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

    st.divider()

    # -----------------------------------------------------------------------
    #  Python connector snippet
    # -----------------------------------------------------------------------
    st.subheader("2. Python connector snippet")
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

    st.divider()

    # -----------------------------------------------------------------------
    #  Copy-into SQL
    # -----------------------------------------------------------------------
    st.subheader("3. Bulk load via Stage (CSV → Snowflake)")
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

    st.divider()

    # -----------------------------------------------------------------------
    #  Sample analytics queries
    # -----------------------------------------------------------------------
    st.subheader("4. Sample analytics queries")
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

    st.divider()

    # -----------------------------------------------------------------------
    #  Architecture diagram (ASCII)
    # -----------------------------------------------------------------------
    st.subheader("5. Architecture overview")
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
        "Once your data is in Snowflake, return to **Power BI Dashboard** mode to download CSVs "
        "or connect Power BI directly using the Snowflake connector."
    )
