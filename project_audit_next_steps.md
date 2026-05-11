# Market Basket Analysis Project Audit

Date checked: 2026-05-11

Repository: https://github.com/omartabanga/Market-Basket-Analysis-

## Current Files

- `Groceries_dataset.csv`: raw transaction data.
- `transaction_matrix.csv`: binary basket matrix.
- `association_rules.csv`: final exported rules.
- `EDA_Notebook.ipynb`: EDA and preprocessing notebook.
- `EDA_Notebook_Milestone3.ipynb`: EDA plus Apriori, FP-Growth, association rules, metrics, and interpretation.
- `README.md`: project overview, but currently behind the notebook progress.

## Dataset Status

- Raw rows: 38,765 purchase records.
- Transaction matrix rows: 14,963 transactions.
- Raw columns: `Member_number`, `Date`, `itemDescription`.
- The matrix is already built in basket format and ready for association-rule mining.

## What Is Already Done

### EDA and preprocessing

- Loaded and inspected the groceries dataset.
- Checked missing values and duplicates.
- Cleaned item names.
- Converted dates and created temporal features.
- Analyzed item frequencies.
- Analyzed rare items.
- Analyzed transaction sizes.
- Built a binary transaction matrix.
- Exported `transaction_matrix.csv`.

### Association-rule mining

The Milestone 3 notebook already includes:

- Apriori implementation using `mlxtend.frequent_patterns.apriori`.
- FP-Growth implementation using `mlxtend.frequent_patterns.fpgrowth`.
- Threshold experiment with support values:
  - 0.005
  - 0.010
  - 0.020
  - 0.050
- Chosen support threshold: `min_support = 0.005`.
- Frequent itemsets:
  - Apriori: 126 itemsets.
  - FP-Growth: 126 itemsets.
  - Itemset lengths: 89 single-itemsets and 37 two-itemsets.
- Association rules generated with `lift >= 1.0`.
- Additional metrics:
  - support
  - confidence
  - lift
  - leverage
  - conviction
  - jaccard
- Exported `association_rules.csv`.

## Current Association Rule Results

Only 6 rules are exported.

Strongest rules by lift:

| Rank | Rule | Lift | Confidence |
|---:|---|---:|---:|
| 1 | other vegetables -> frankfurter | 1.1161 | 0.0421 |
| 2 | frankfurter -> other vegetables | 1.1161 | 0.1363 |
| 3 | yogurt -> sausage | 1.1090 | 0.0669 |
| 4 | sausage -> yogurt | 1.1090 | 0.0952 |
| 5 | soda -> sausage | 1.0150 | 0.0613 |
| 6 | sausage -> soda | 1.0150 | 0.0986 |

Important interpretation:

- There are no rules with `lift > 1.2`.
- There are no rules with `confidence > 0.5`.
- This is likely because baskets are small and sparse, not necessarily because the implementation is wrong.
- Average baskets are around 2.5 to 2.6 items.

## Important Problems / Gaps

### 1. README is outdated

The README still says these are "Next Steps":

- Apply Apriori.
- Apply FP-Growth.
- Generate and evaluate association rules.

But the Milestone 3 notebook already does these things. The README should be updated.

### 2. README references missing images

The README points to files under `assets/`, for example:

- `assets/01_top20_items.png`
- `assets/08_wordcloud.png`
- `assets/09_cooccurrence_heatmap.png`

But the repository currently has no `assets/` directory. The images may display inside the notebook, but they are not exported as repo files.

### 3. Final rules are too few for a strong final project

The current final output has only 6 rules. That may be acceptable if explained well, but a final project usually needs deeper experimentation.

Recommended improvement:

- Try multiple support/confidence/lift settings.
- Compare rule counts and quality.
- Explain why stricter thresholds produce few or zero rules.
- Consider using lower support with caution, then filter by lift and business usefulness.

### 4. Recommendation engine is not implemented as a reusable function

The README says recommendation engine is a next step. The notebook has business recommendations, but not a clear function like:

```python
recommend_products(["sausage"], rules, top_n=5)
```

This is a good part to implement.

### 5. Customer segmentation is not implemented

The README lists customer segmentation as a next step, but no clear segmentation section exists.

This could be done using:

- customer-level product category vectors
- purchase frequency
- basket size behavior
- KMeans clustering

### 6. Final report / presentation structure is missing

The repo needs a final-story structure:

- business problem
- dataset description
- preprocessing
- EDA insights
- Apriori vs FP-Growth
- final rules
- recommendations
- limitations
- future work

## Best Part for You to Own

Based on the current repo, the best high-impact part for one person to own is:

### Recommendation engine + final interpretation

Why this is good:

- It builds directly on the existing association rules.
- It is not just repeating EDA.
- It gives the project a practical business output.
- It is easy to explain in the presentation.

Suggested tasks:

1. Build a recommendation function from association rules.
2. Allow input product(s), then return recommended products.
3. Rank recommendations by lift, confidence, and support.
4. Add examples:
   - input: `frankfurter`
   - output: `other vegetables`
   - input: `yogurt`
   - output: `sausage`
   - input: `sausage`
   - output: `yogurt`, `soda`
5. Add a short business explanation for each recommendation.
6. Add a final limitations section explaining low confidence and sparse baskets.

## Alternative Part to Own

### Improve evaluation and reporting

Tasks:

1. Update README so it matches the real notebook.
2. Export missing visualizations into `assets/`.
3. Add a final summary table for:
   - support threshold
   - frequent itemset count
   - rule count
   - max lift
   - max confidence
   - runtime
4. Add a final project report section.

## What I Need Next

To continue accurately, provide:

1. The official project requirements or rubric.
2. The exact part assigned to you.
3. The dataset you agreed on if it is different from `Groceries_dataset.csv`.
4. The playlist/reference links.
5. Whether you want the output as:
   - notebook changes
   - README/report
   - presentation
   - all of them
