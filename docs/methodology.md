# Design and evaluation protocol

1. Validate the source with reusable Pandas `.pipe` stages; remove 845 empty reviews and eight duplicate records, retaining 22,633 reviews.
2. Separate products across training (14,158), validation (2,762) and test (5,713) reviews. Run three-fold product-group cross-validation inside training only.
3. Compare random undersampling at two ratios, random oversampling, SMOTENC and balanced class weights against an unbalanced reference. Resampling occurs only inside fitted training pipelines.
4. Compare mean, median, KNN and iterative numeric imputation under a reproducible 20% MCAR stress simulation. Source numeric fields have no missing values; category missingness is preserved.
5. Fit an additional full-text TF–IDF reference, choose each scenario's winner by validation average precision and choose F1 thresholds on validation. Freeze those choices before test scoring.
6. Serialize the entire selected preprocessing/classification pipeline and verify identical probabilities after reload.

## Findings

| Pipeline | Test average precision | Test log loss |
|---|---:|---:|
| Full-text TF–IDF reference | 0.795 | 0.221 |
| Compressed text, no resampling | 0.636 | 0.282 |
| Compressed text, SMOTENC | 0.627 | 0.374 |
| Compressed text, balanced weights | 0.632 | 0.415 |

Preserving lexical information mattered more than balancing classes in this experiment. The resampling comparison holds the compressed representation fixed; the full-text reference changes representation and is not evidence that resampling caused the difference. Equalizing class counts worsened log loss and shifted decision thresholds. Iterative imputation won the missingness scenario with AP 0.631, but all four imputers were close (0.626–0.631).

The label is a recommendation decision, not a verified complaint. English-language reviews, a single retail corpus and MCAR simulation limit transfer to live workflows. Human review, cost-based threshold selection and monitoring remain necessary before real deployment.

## Evidence contract

Committed CSV/NPZ files are actual run outputs. Independent verification recomputes metrics or attribution aggregates, checks source hashes and confirms notebook execution. Model selection never uses the final test results. Read source modules for the exact numerical definitions; the reports are intentionally small enough to inspect locally.

## Technical references

- [Imbalanced-learn sampling guide](https://imbalanced-learn.org/stable/over_sampling.html)
