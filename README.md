# Market-Basket-Analysis
<p align="center">
  <img src="assets/08_wordcloud.png" alt="Product Word Cloud" width="800"/>
</p>

<h1 align="center">🛒 Groceries Market Basket Analysis</h1>

<p align="center">
  <strong>Exploratory Data Analysis &amp; Preprocessing for Association Rule Mining</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Matplotlib-3.x-11557C" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Seaborn-0.13-4C72B0" alt="Seaborn"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

---

## 📌 Project Overview

This project performs a comprehensive **Exploratory Data Analysis (EDA)** on a real-world grocery transactions dataset containing **38,765 purchase records** from nearly 4,000 members over a 2-year period (2014–2015). The goal is to understand purchasing patterns, clean and preprocess the data, and produce a **binary transaction matrix** ready for association rule mining algorithms like **Apriori** and **FP-Growth**.

---

## 📊 Dataset at a Glance

| Metric | Value |
|:--|--:|
| Total purchase records | 38,765 |
| Unique members | 3,898 |
| Unique products | 167 |
| Unique transactions | 14,963 |
| Date range | Jan 2014 – Dec 2015 |

The raw data consists of three columns: `Member_number`, `Date`, and `itemDescription`. Each row represents one item purchased by a member on a given date.

---

## 🔍 What Was Done

### 1. Data Exploration

- Profiled the dataset for shape, types, missing values, and duplicates.
- Analyzed **product frequency distributions** — the catalog follows a power-law: a handful of staples (whole milk, vegetables, rolls/buns) dominate while a long tail of 13 rare items appear fewer than 10 times.
- Computed **transaction sizes** — baskets average 2.6 items with a max of 11.
- Examined **seasonal and temporal trends** — monthly volume is remarkably stable (~1,400–1,600/month) with no strong weekday/weekend or seasonal effects.

### 2. Data Preprocessing

- **Cleaned item descriptions** — stripped whitespace, lowercased for consistency.
- **Scanned for cancelled/invalid transactions** — none found in this dataset.
- **Parsed dates** — converted DD-MM-YYYY strings to datetime, extracted year/month/day-of-week features.
- **Removed rare items** — filtered out 13 products with <10 purchases (7.8% of catalog, 0.2% of rows) to reduce noise.
- **Built a binary transaction matrix** — transformed the long-format data into a 14,963 × 154 basket matrix with 98.35% sparsity, ready for association algorithms.

### 3. Visualizations

Created 10 publication-quality charts covering every dimension of the data:

| # | Visualization | Insight |
|:-:|:--|:--|
| 1 | Top 20 items bar chart | Whole milk leads with 2,502 purchases |
| 2 | Rare items bar chart | 13 items with <10 purchases flagged as noise |
| 3 | Item frequency distribution | Confirms power-law / long-tail shape |
| 4 | Transaction size histogram | Heavy concentration at 2–3 items per basket |
| 5 | Monthly purchase trend | Stable volume over 24 months |
| 6 | Day-of-week bar chart | Nearly uniform distribution across all days |
| 7 | Monthly seasonality chart | Mild peaks in May and August |
| 8 | Product word cloud | Visual summary of catalog dominance |
| 9 | Co-occurrence heatmap | Whole milk + other vegetables is the top pair (222) |
| 10 | Transaction matrix sparsity | Visual confirmation of expected sparse structure |

---

## 📈 Key Visualizations

<details>
<summary><b>🏆 Top 20 Products</b></summary>
<br/>
<img src="assets/01_top20_items.png" alt="Top 20 Items" width="700"/>
</details>

<details>
<summary><b>📉 Item Frequency Distribution</b></summary>
<br/>
<img src="assets/03_item_freq_distribution.png" alt="Frequency Distribution" width="700"/>
</details>

<details>
<summary><b>🧺 Transaction Size Distribution</b></summary>
<br/>
<img src="assets/04_transaction_sizes.png" alt="Transaction Sizes" width="700"/>
</details>

<details>
<summary><b>📅 Monthly Purchase Trend</b></summary>
<br/>
<img src="assets/05_monthly_trend.png" alt="Monthly Trend" width="700"/>
</details>

<details>
<summary><b>🔥 Product Co-occurrence Heatmap</b></summary>
<br/>
<img src="assets/09_cooccurrence_heatmap.png" alt="Co-occurrence Heatmap" width="700"/>
</details>

<details>
<summary><b>🗓️ Day of Week &amp; Monthly Seasonality</b></summary>
<br/>
<img src="assets/06_day_of_week.png" alt="Day of Week" width="400"/>
<img src="assets/07_monthly_seasonality.png" alt="Monthly Seasonality" width="400"/>
</details>

---



## 🔑 Key Findings

1. **Whole milk is king** — purchased 2,502 times, appearing in ~17% of all transactions.
2. **Power-law catalog** — the top 10 items account for 32% of all purchases; 13 items have fewer than 10 purchases total.
3. **Small baskets** — average transaction contains only 2.6 items, with most baskets holding exactly 2.
4. **No seasonality** — monthly and day-of-week volumes are remarkably stable, meaning association rules should generalize across time.
5. **Strongest pair** — whole milk and other vegetables co-occur 222 times, making them the top candidate for association rules.
6. **Sparse matrix** — the final transaction matrix is 98.35% sparse, which is typical for market basket data and well-suited for efficient sparse-matrix association algorithms.

---

## 🔮 Next Steps

- [ ] Apply **Apriori algorithm** with varying support/confidence thresholds
- [ ] Apply **FP-Growth** for faster frequent itemset mining
- [ ] Generate and evaluate **association rules** (lift, confidence, conviction)
- [ ] Build a **product recommendation engine** from strong rules
- [ ] Perform **customer segmentation** based on purchasing patterns

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation and transaction matrix construction
- **matplotlib** / **seaborn** — statistical visualizations
- **wordcloud** — product name word cloud generation
- **numpy** — numerical operations

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built as part of a Market Basket Analysis pipeline for association rule mining.</i>
</p>
