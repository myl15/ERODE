# Significance Testing Methods

This document defines the inferential protocol used for judge-output comparisons in SycoBench.

## Reporting Policy

- Primary evidence: effect sizes and confidence intervals.
- Secondary evidence: multiplicity-corrected p-values (Benjamini-Hochberg FDR).
- Recommended language in paper text:
  - "effect size and 95% CI indicate ..."
  - "FDR-adjusted p-value is reported as secondary support"

## Comparison Families

- Main table family: baseline judge vs every other judge.
- Appendix family: all pairwise judge comparisons.
- FDR is computed separately for each family.

## Outcomes Tested

- Binary outcome: `sycophancy_rate`
  - Effect size: risk difference (`rate_a - rate_b`)
  - Additional diagnostic effect: log odds ratio (with Haldane-Anscombe 0.5 correction)
  - CI for proportions in metric summaries: Wilson score interval (`sycophancy_rate_wilson_ci`)
  - CI for cross-model effect size (`risk_difference`): percentile bootstrap CI
  - p-value: two-sided permutation test on mean difference of Bernoulli outcomes
- Continuous outcomes: `ERS`, `hedge_drift`
  - Effect size: mean difference (`mean_a - mean_b`)
  - Additional standardized effect: Cohen's d
  - CI: percentile bootstrap CI for mean difference
  - p-value: two-sided permutation test on mean difference

## Stratification

Tests are run for:

- overall pooled
- per domain
- per pressure type
- domain x pressure cells

## Minimum Sample Rules

- Binary metric minimum sample (`n_eligible`) per side: `min_n_binary = 5`
- Continuous metric minimum sample per side: `min_n_continuous = 8`
- If either side is below threshold, the result is flagged `insufficient_data: true` and inferential values are not interpreted.

## Reproducibility Settings

- Bootstrap resamples: `n_bootstrap = 2000`
- Permutations: `n_permutations = 2000`
- Seed: `7`
- Confidence level: `0.95`
- Nominal alpha: `0.05`

## Output Artifacts

Running the standard pipeline command writes:

- `analysis/significance_summary.json` (full structured output + metadata)
- `analysis/significance_main_baseline_vs_others.json` (main-paper family)
- `analysis/significance_appendix_pairwise.json` (appendix family)

The significance summary also includes Fisher exact analyses with Bonferroni correction:

- `results.fisher_domain_pairwise_per_model`
  - Domain pairwise Fisher tests (within each model):
    - `established_fact` vs `contested_claim`
    - `established_fact` vs `subjective_judgment`
    - `contested_claim` vs `subjective_judgment`
- `results.fisher_claude_vs_others_contested_claim`
  - Fisher tests for `claude_sonnet46` vs each other model on contested-claim sycophancy

## Reproducibility Command

```bash
uv run python generate_visuals.py gpt4o claude_sonnet46 llama3_70b gemini25flash
```

Replace model keys as needed. The first model key is treated as the baseline by default.
