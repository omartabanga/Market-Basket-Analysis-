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
    page_title="Market Basket Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
#  Professional Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0;
        font-size: 13px;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* Main content background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero header area */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero h1 {
        color: #f8fafc;
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #94a3b8;
        font-size: 15px;
        margin: 0;
        font-weight: 400;
    }

    /* KPI Cards row */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 32px;
    }
    .kpi-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04), 0 2px 4px -2px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .kpi-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .kpi-change {
        font-size: 12px;
        color: #059669;
        font-weight: 500;
        margin-top: 8px;
    }

    /* Section cards */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #0ea5e9 0%, #0284c7 100%);
        border-radius: 2px;
    }

    /* Info / tip boxes */
    .tip-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
        font-size: 13px;
        color: #1e40af;
        line-height: 1.6;
    }
    .tip-box strong {
        color: #1e3a5f;
    }
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #22c55e;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
        font-size: 13px;
        color: #166534;
        line-height: 1.6;
    }
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
        font-size: 13px;
        color: #92400e;
        line-height: 1.6;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(14,165,233,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 12px -2px rgba(14,165,233,0.3);
    }

    /* Tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        color: #64748b;
        border: none;
        background: transparent;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
    }

    /* Sliders */
    .stSlider label {
        font-size: 12px;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Multiselect */
    .stMultiSelect label {
        font-size: 12px;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Sidebar header */
    .sidebar-title {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .sidebar-desc {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.5;
        margin-bottom: 20px;
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
def kpi_card(label: str, value: str, change: str = "") -> None:
    change_html = f'<div class="kpi-change">{change}</div>' if change else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            {change_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, content_func) -> None:
    st.markdown(f'<div class="section-card"><div class="section-title">{title}</div>', unsafe_allow_html=True)
    content_func()
    st.markdown('</div>', unsafe_allow_html=True)


def tip_box(content: str) -> None:
    st.markdown(f'<div class="tip-box">{content}</div>', unsafe_allow_html=True)


def success_box(content: str) -> None:
    st.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)


def warning_box(content: str) -> None:
    st.markdown(f'<div class="warning-box">{content}</div>', unsafe_allow_html=True)


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
#  Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Platform Selector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-desc">Choose the analytics experience you want to explore.</div>', unsafe_allow_html=True)

    dashboard_mode = st.radio(
        label="",
        options=[
            "Market Basket Engine",
            "Power BI Preview",
            "Snowflake Cloud",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-desc">'
        '<strong>Market Basket Engine</strong> &mdash; Interactive rules and recommendations.<br><br>'
        '<strong>Power BI Preview</strong> &mdash; Live dashboard preview and export tools.<br><br>'
        '<strong>Snowflake Cloud</strong> &mdash; Cloud warehouse setup guide.'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
#  Load data
# ---------------------------------------------------------------------------
raw_data = load_raw_data()
rules = load_rules()
matrix_items = load_matrix_columns()
transaction_items = build_transaction_items(raw_data)


# =============================================================================
#  MODE 1 - Market Basket Engine
# =============================================================================
if dashboard_mode == "Market Basket Engine":
    st.markdown(
        '<div class="hero">'
        '<h1>Market Basket Analysis</h1>'
        '<p>Groceries transaction analysis, association rules, and product recommendations</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_rules, tab_recommender, tab_segments, tab_deploy = st.tabs(
        ["Overview", "Rules Explorer", "Recommendation Engine", "Basket Segments", "Deployment"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Purchase Records", f"{len(raw_data):,}")
        with c2:
            kpi_card("Unique Transactions", f"{raw_data['transaction_id'].nunique():,}")
        with c3:
            kpi_card("Unique Products", f"{raw_data['itemDescription'].nunique():,}")
        with c4:
            kpi_card("Association Rules", f"{len(rules):,}")

        def top_products_content():
            top_n = st.slider("Number of products to display", min_value=5, max_value=30, value=15, key="top_products")
            top_items = raw_data["itemDescription"].value_counts().head(top_n)
            st.bar_chart(top_items, use_container_width=True)

        section_card("Top Products", top_products_content)

        col_a, col_b = st.columns(2)
        with col_a:
            def monthly_content():
                monthly = raw_data.groupby("month").size()
                st.line_chart(monthly, use_container_width=True)
            section_card("Monthly Purchase Volume", monthly_content)
        with col_b:
            def size_content():
                size_counts = transaction_items.apply(len).value_counts().sort_index()
                st.bar_chart(size_counts, use_container_width=True)
            section_card("Transaction Size Distribution", size_content)

    with tab_rules:
        def rules_filter_content():
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
                st.subheader("Rules Ranked by Lift")
                lift_chart = filtered_rules.set_index("Rule")["Lift"] if not filtered_rules.empty else pd.Series(dtype=float)
                st.bar_chart(lift_chart, use_container_width=True)
            with col_b:
                st.subheader("Metric Summary")
                if filtered_rules.empty:
                    st.info("No rules match the selected filters.")
                else:
                    st.dataframe(
                        filtered_rules[["Support", "Confidence", "Lift", "Leverage", "Conviction", "Jaccard"]]
                        .describe()
                        .round(4),
                        use_container_width=True,
                    )

            tip_box(
                "<strong>How to read these rules:</strong> Lift above 1 means the antecedent and consequent "
                "appear together more often than expected under independence. The current dataset has small, "
                "sparse baskets, so confidence values are generally low."
            )

        section_card("Association Rules Explorer", rules_filter_content)

    with tab_recommender:
        def recommender_content():
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
                    warning_box("No rule-based recommendation found. Showing co-occurrence fallback.")
                    fallback = fallback_cooccurrence(selected_items, transaction_items, top_n_recs)
                    if fallback.empty:
                        st.info("No co-occurring products found for this basket.")
                    else:
                        st.dataframe(
                            fallback.style.format(
                                {"Basket match rate": "{:.2%}", "Dataset support": "{:.2%}"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    success_box(f"Found <strong>{len(recommendations)}</strong> recommendation(s) based on your basket.")
                    st.dataframe(
                        recommendations.style.format(
                            {"Support": "{:.4f}", "Confidence": "{:.4f}", "Lift": "{:.4f}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("Select at least one product to generate recommendations.")

            st.subheader("Ready-Made Examples")
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

        section_card("Product Recommendation Engine", recommender_content)

    with tab_segments:
        def segments_content():
            segments = basket_segments(transaction_items)
            st.dataframe(
                segments.style.format({"average_size": "{:.2f}", "share": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(segments.set_index("segment")["transactions"], use_container_width=True)

            tip_box(
                "<strong>Insight:</strong> Most baskets are small (1-3 items), which explains why association "
                "rules have low confidence. Low confidence is a data characteristic here, not a failed model."
            )

        section_card("Basket Size Segmentation", segments_content)

    with tab_deploy:
        def deploy_content():
            st.markdown(
                """
                1. Push all project files to GitHub (`app.py`, `requirements.txt`, `.streamlit/config.toml`, and the CSV files).
                2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in with GitHub.
                3. Click **New app** and select this repository.
                4. Set the main file path to `app.py`.
                5. Click **Deploy**.
                """
            )

        section_card("Deployment Checklist", deploy_content)


# =============================================================================
#  MODE 2 - Power BI Preview
# =============================================================================
elif dashboard_mode == "Power BI Preview":
    st.markdown(
        '<div class="hero">'
        '<h1>Power BI Dashboard Preview</h1>'
        '<p>Live preview of your basket analytics and export tools for Power BI Desktop</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("What is this page and how do I use it?", expanded=True):
        st.markdown(
            """
            This page does **two things**:

            **1. Live Preview Dashboard** - See your data visualized with interactive charts 
            directly in the browser. This is a preview of what your Power BI report could look like.

            **2. Export and Connect** - Download the cleaned data as CSV/Excel so you can 
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

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Records", f"{len(raw_data):,}")
    with c2:
        kpi_card("Transactions", f"{raw_data['transaction_id'].nunique():,}")
    with c3:
        kpi_card("Products", f"{raw_data['itemDescription'].nunique():,}")
    with c4:
        kpi_card("Avg Basket Size", f"{transaction_items.apply(len).mean():.1f}")

    # Charts row 1
    col_a, col_b = st.columns(2)
    with col_a:
        def top15_content():
            top_15 = raw_data["itemDescription"].value_counts().head(15)
            st.bar_chart(top_15, use_container_width=True)
        section_card("Top 15 Products", top15_content)
    with col_b:
        def monthly_tx_content():
            monthly_tx = raw_data.groupby("month")["transaction_id"].nunique()
            st.line_chart(monthly_tx, use_container_width=True)
        section_card("Monthly Transaction Volume", monthly_tx_content)

    # Charts row 2
    col_c, col_d = st.columns(2)
    with col_c:
        def scatter_content():
            if not rules.empty:
                scatter_data = rules[["Confidence", "Lift", "Support", "Rule"]].copy()
                scatter_data["Size"] = scatter_data["Support"] * 5000
                st.scatter_chart(
                    scatter_data,
                    x="Confidence",
                    y="Lift",
                    size="Size",
                    color="Support",
                    use_container_width=True,
                )
        section_card("Rules: Confidence vs Lift", scatter_content)
    with col_d:
        def seg_content():
            seg = basket_segments(transaction_items)
            st.bar_chart(seg.set_index("segment")["transactions"], use_container_width=True)
        section_card("Basket Segments", seg_content)

    # Rules table
    def rules_table_content():
        top_rules = rules.head(20)[["Rank", "Rule", "Support", "Confidence", "Lift"]]
        st.dataframe(
            top_rules.style.format({"Support": "{:.4f}", "Confidence": "{:.4f}", "Lift": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
            height=300,
        )
    section_card("Top Association Rules", rules_table_content)

    st.divider()

    # Export section
    def export_content():
        st.write("Download these files and import them into Power BI Desktop using **Get Data > Text/CSV**.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                label="Download Raw transactions (CSV)",
                data=raw_data.to_csv(index=False).encode("utf-8"),
                file_name="groceries_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                label="Download Association rules (CSV)",
                data=rules.to_csv(index=False).encode("utf-8"),
                file_name="association_rules.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c3:
            matrix_df = pd.read_csv(MATRIX_PATH)
            st.download_button(
                label="Download Transaction matrix (CSV)",
                data=matrix_df.to_csv(index=False).encode("utf-8"),
                file_name="transaction_matrix.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.download_button(
            label="Download All rules + metrics (Excel)",
            data=to_excel(rules, "AssociationRules"),
            file_name="association_rules.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    section_card("Export Data for Power BI Desktop", export_content)

    st.divider()

    # DAX measures
    def dax_content():
        st.write("Copy these formulas into Power BI Desktop under **Modeling > New Measure**.")
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
    section_card("DAX Measures for Power BI", dax_content)

    st.divider()

    # Embed section
    def embed_content():
        tip_box(
            "<strong>How to get your Embed URL:</strong><br>"
            "1. Build your report in <strong>Power BI Desktop</strong>.<br>"
            "2. Click <strong>Publish</strong> and sign in to your Power BI account.<br>"
            "3. Go to <strong>Power BI Service</strong> (app.powerbi.com) and open your published report.<br>"
            "4. Click <strong>File > Embed report > Website or portal</strong>.<br>"
            "5. Copy the URL that looks like <code>https://app.powerbi.com/reportEmbed?reportId=...</code><br>"
            "6. Paste it below."
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
    section_card("Embed Live Power BI Report", embed_content)


# =============================================================================
#  MODE 3 - Snowflake Cloud
# =============================================================================
elif dashboard_mode == "Snowflake Cloud":
    st.markdown(
        '<div class="hero">'
        '<h1>Snowflake Data Cloud Setup</h1>'
        '<p>Warehouse schema, connection code, and SQL scripts for cloud analytics</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tip_box(
        "<strong>Why Snowflake?</strong> Snowflake gives you elastic compute, automatic scaling, and the "
        "ability to run SQL analytics or connect Power BI / Tableau directly to your warehouse. "
        "You can sign up for a <a href='https://signup.snowflake.com/' target='_blank'>free 30-day trial</a> "
        "with $400 in credits."
    )

    tab_sql, tab_python, tab_copy, tab_queries, tab_arch = st.tabs(
        ["SQL Schema", "Python Loader", "Bulk Load", "Analytics Queries", "Architecture"]
    )

    with tab_sql:
        def sql_content():
            sql_ddl = """
CREATE DATABASE IF NOT EXISTS market_basket_db;
CREATE SCHEMA IF NOT EXISTS market_basket_db.analytics;
USE SCHEMA market_basket_db.analytics;

CREATE OR REPLACE TABLE groceries_raw (
    Member_number   INTEGER,
    Date            DATE,
    itemDescription VARCHAR(255),
    transaction_id  VARCHAR(50),
    month           VARCHAR(7),
    day_name        VARCHAR(20)
);

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

CREATE OR REPLACE TABLE transaction_matrix (
    transaction_id VARCHAR(50)
);
"""
            st.code(sql_ddl, language="sql")
        section_card("SQL DDL - Create Tables", sql_content)

    with tab_python:
        def py_content():
            python_snippet = """
import snowflake.connector
import pandas as pd

conn = snowflake.connector.connect(
    account="YOUR_ACCOUNT",
    user="YOUR_USER",
    password="YOUR_PASSWORD",
    warehouse="COMPUTE_WH",
    database="market_basket_db",
    schema="analytics",
    role="ACCOUNTADMIN",
)

raw_data = pd.read_csv("Groceries_dataset.csv")
success, nchunks, nrows, _ = conn.write_pandas(
    raw_data, "groceries_raw", overwrite=True
)
print(f"Loaded {nrows} rows into Snowflake.")
"""
            st.code(python_snippet, language="python")
            st.info(
                "In production, never hard-code credentials. Use Streamlit secrets or environment variables."
            )
        section_card("Python Connector Snippet", py_content)

    with tab_copy:
        def copy_content():
            copy_sql = """
COPY INTO groceries_raw
FROM @my_stage/groceries_cleaned.csv
FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1, FIELD_OPTIONALLY_ENCLOSED_BY = '"');

COPY INTO association_rules
FROM @my_stage/association_rules.csv
FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1, FIELD_OPTIONALLY_ENCLOSED_BY = '"');
"""
            st.code(copy_sql, language="sql")
        section_card("Bulk Load via Stage (CSV to Snowflake)", copy_content)

    with tab_queries:
        def queries_content():
            analytics_sql = """
SELECT itemDescription, COUNT(*) AS purchases
FROM groceries_raw
GROUP BY itemDescription
ORDER BY purchases DESC
LIMIT 20;

SELECT month, COUNT(DISTINCT transaction_id) AS transactions
FROM groceries_raw
GROUP BY month
ORDER BY month;

SELECT Antecedent, Consequent, Lift, Confidence, Support
FROM association_rules
WHERE Lift > 1.0
ORDER BY Lift DESC;

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
        section_card("Sample Analytics Queries", queries_content)

    with tab_arch:
        def arch_content():
            st.text(
                """
    +-----------------+        +------------------+        +------------------+
    |  CSV Data       |  PUT   |  Snowflake       |  SQL   |  Power BI /      |
    |  (Groceries)    | -----> |  Stage > Tables  | -----> |  Tableau /       |
    +-----------------+        +------------------+        |  Streamlit       |
                                                          +------------------+
    
    Benefits:
    - Elastic storage (no local .csv bloat)
    - Concurrent read/write without locking
    - Direct Power BI connector (Import or DirectQuery)
    - Scales to billions of rows automatically
                """
            )
        section_card("Architecture Overview", arch_content)

    st.divider()
    success_box(
        "Once your data is in Snowflake, return to <strong>Power BI Preview</strong> mode to download CSVs "
        "or connect Power BI directly using the Snowflake connector."
    )
