# Beyond Territorial Gaps – analysis code

Code to reproduce all tables, figures and statistics in:

> Vidal-Sepúlveda, M., Beroiza, F., & Olivares-Rodríguez, C. (in review). *Beyond Territorial Gaps: Attitudes, Skills, and Artificial Intelligence Adoption Profiles among Rural Primary Teachers in Chile*. 

The data are archived separately on Zenodo: 10.5281/zenodo.24555627.

## Requirements

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

## Usage

1. The dataset is available from the authors upon reasonable request.
2. Run:

```bash
python code/reproduce_analysis.py --data data/ai_rural_teachers_chile_full.csv --out output
```

The script takes about 30 seconds. `--boot` sets the number of wild cluster bootstrap replications (default 9,999).

## Outputs

| File | Article |
|---|---|
| `table3_descriptives_by_region.csv` | Table 3, regional comparisons, power and equivalence tests |
| `table4_variance_decomposition.csv` | Table 4 and Figure 1 |
| `table5_ols_models.csv` | Table 5 (Model 1, Model 2 and the sensitivity model with intention) |
| `table6_profiles.csv` | Table 6 |
| `table7_firth_models.csv` | Table 7 |
| `school_units.csv` | Figure 2 |
| `figure1_variance_decomposition.png`, `figure2_ai_use_by_school.png` | Figures 1 and 2 |
| `results.json` | Other statistics cited in the text (reliability, power, diagnostics, profile solutions, robustness check) |

## Analyses

1. Regional comparisons: Welch's *t*, Mann–Whitney *U*, Cohen's *d*, sensitivity power analysis and TOST equivalence tests (±0.3 and ±0.5 *SD*).
2. Variance decomposition into region, school units within region and teachers within school units; *F* tests and ICC(1).
3. OLS regression of standardised AI use with HC3 standard errors and wild cluster restricted bootstrap *p*-values (Webb six-point weights, nine school units).
4. Profiles of AI use: partitioning around medoids on the three standardised use items (best of 50 random starts), silhouette widths for 2–4 profiles, agreement with *k*-means.
5. Firth-penalised logistic regression of non-adopter profile membership, with profile-likelihood confidence intervals and penalised likelihood-ratio tests.
6. Robustness check: profiles derived from attitudinal and skill measures.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0 (see the Zenodo record).

## Funding

This work was funded by the Chilean Undersecretariat of Education through the Education Research and Development Fund (FONIDE, Project FON2400113, 2024).