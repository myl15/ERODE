# ERODE

**ERODE** (Epistemic Robustness under Oppositional Dialogue Evaluation) is a research benchmark and analysis pipeline for measuring how **instruction-tuned LLMs** respond when users apply **social and rhetorical pressure** across **knowledge domains** (established facts, contested claims, subjective judgments). The project combines **multi-turn scripted dialogues**, an **independent LLM judge**, **lexical features** (hedging, boosters), and **publication-style figures** for papers and talks.

---

## Highlights (conference paper bundle)

Headline numbers, figures, and **inferential outputs** for the paper use the **GLM-5.1 judge** snapshot in [`analysis/glm_z1_judged/`](analysis/glm_z1_judged/) ([`metrics_summary.json`](analysis/glm_z1_judged/metrics_summary.json), [`significance_summary.json`](analysis/glm_z1_judged/significance_summary.json), plus the split files [`significance_main_baseline_vs_others.json`](analysis/glm_z1_judged/significance_main_baseline_vs_others.json) and [`significance_appendix_pairwise.json`](analysis/glm_z1_judged/significance_appendix_pairwise.json)). Evaluated targets: **Claude Sonnet 4.6**, **Llama 3 70B**, **Gemini 2.5 Flash**, and **GPT-4o** (same keys as in the metrics JSON).

Methods for permutation tests, bootstrap CIs, Benjamini–Hochberg FDR, and Fisher exact tests are documented in [`analysis/SIGNIFICANCE_METHODS.md`](analysis/SIGNIFICANCE_METHODS.md).

### Main inferential claims (this judge, four models)

1. **Claude vs every other model (overall sycophancy rate).** Claude has a **significantly lower** pooled sycophancy rate than Llama 3 70B, Gemini 2.5 Flash, and GPT-4o. In `significance_summary.json`, the **baseline-vs-others** family uses **Claude as `model_a`**; for `metric == "sycophancy_rate"` and `scope_key == "overall"`, all three comparisons have **FDR-adjusted *p* ≈ 0.0066** (α = 0.05), with bootstrap 95% CIs on the risk difference (Claude minus other) strictly **below zero** (roughly **−27 to −11 points** vs Llama, **−22 to −6.5** vs Gemini, **−20 to −5.6** vs GPT-4o). Point estimates: Claude **~2.1%** vs **~20.7%** / **~15.9%** / **~14.7%** on the other three.

2. **Contested-claim domain vs other domains (within-model Fisher, Bonferroni).** For **Gemini**, **GPT-4o**, and **Llama**, **contested-claim** scenarios carry the **highest** pointwise sycophancy rate among the three knowledge domains. **Claude** is an edge case (0% on both established-fact and contested-claim eligible dialogues in this run, with a small nonzero rate only on subjective judgment), but **Claude vs each other model on contested claims alone** still shows a large gap: Fisher exact **Bonferroni-adjusted *p* < 0.002** for every pairwise model contrast (Claude **0 / 46** vs **7–15 / 28–37** on the others).

3. **Domain gradient (established fact → contested claim → subjective judgment).** The paper reads a **directionally consistent** gradient along this epistemic axis: **all four models** are weakest on **established-fact** scenarios (near-zero SR for Claude; low single digits for Llama and GPT-4o; still lowest for Gemini), with substantially higher rates in the **contested** and **subjective** strata. In raw point estimates, **Gemini**, **GPT-4o**, and **Llama** peak on **contested claims** rather than subjective judgment; **Claude** is mostly flat at zero until subjective. **Multiplicity-adjusted domain Fisher tests** (`results.fisher_domain_pairwise_per_model` in `significance_summary.json`) nevertheless reject equality only for **Llama 3 70B** (Bonferroni-significant for established vs contested and established vs subjective; contested vs subjective **not** significant). **Gemini** and **GPT-4o** do **not** clear Bonferroni at α = 0.05; **Claude** has too few events for domain contrasts to be informative.

| Metric (pooled) | Claude Sonnet 4.6 | Llama 3 70B | Gemini 2.5 Flash | GPT-4o |
|-----------------|-------------------|-------------|-------------------|--------|
| **Overall sycophancy rate** | ~2.1% | ~20.7% | ~15.9% | ~14.7% |
| **Overall ERS** (higher = better) | ~4.81 | ~3.92 | ~4.11 | ~4.26 |

**By domain — sycophancy rate (eligible T1 → capitulation at T5):**

| Domain | Claude | Llama | Gemini | GPT-4o |
|--------|--------|-------|--------|--------|
| Established fact | ~0% | ~2% | ~7% | ~2% |
| Contested claim | ~0% | ~41% | ~25% | ~24% |
| Subjective judgment | ~7% | ~26% | ~20% | ~21% |

Re-run [`generate_visuals.py`](generate_visuals.py) (and the upstream dialogue / feature / judge steps) to refresh numbers; by default artifacts land under [`analysis/`](analysis/) unless you configure a different output directory.

For definitions of **SR**, **ERS**, **hedge drift**, and eligibility, see [**Figure guide**](analysis/FIGURE_GUIDE.md). For a **~3 minute talk outline** with the same numbers, see [**Presentation notes**](analysis/PRESENTATION_3MIN.md) (update figures if you switch judge bundles).

---

## Repository layout

| Path | Role |
|------|------|
| [`scenarios/`](scenarios/) | Scenario JSON (question, ground truth, user claim, pressure variants). |
| [`run_dialogues.py`](run_dialogues.py) | Run model dialogues → `transcripts/<model_key>.jsonl`. |
| [`extract_features.py`](extract_features.py) | Lexical + heuristic features → `features/<model_key>_features.jsonl`. |
| [`judge.py`](judge.py) | LLM-as-judge rubrics and position labels → `judgments/<model_key>_judgments.jsonl`. |
| [`metrics.py`](metrics.py) | Aggregate metrics → consumed by `generate_visuals.py`. |
| [`analyze.py`](analyze.py) | Matplotlib figures. |
| [`generate_visuals.py`](generate_visuals.py) | Writes `analysis/metrics_summary.json` and all figures. |
| [`analysis/`](analysis/) | Figures, metrics JSON, significance JSON, and documentation (`FIGURE_GUIDE.md`, `PRESENTATION_3MIN.md`). Subfolder [`analysis/glm_z1_judged/`](analysis/glm_z1_judged/) is the GLM-5.1 judge bundle used in the paper. |
| [`config.py`](config.py) | Model registry, paths, judge model config. |

---

## Setup

Requires **Python ≥ 3.12** and [`uv`](https://github.com/astral-sh/uv) (recommended).

```bash
uv sync --extra dev
```

Set API keys as needed for models you run (see `config.py`):

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TOGETHER_API_KEY`, etc.

---

## Pipeline (end-to-end)

1. **Dialogues** — run scenarios for a model key defined in `config.MODELS`:

   ```bash
   uv run python run_dialogues.py gemini25flash
   ```

2. **Features**

   ```bash
   uv run python extract_features.py gemini25flash
   ```

3. **Judgments**

   ```bash
   uv run python judge.py gemini25flash
   ```

4. **Metrics and figures**

   ```bash
   uv run python generate_visuals.py claude_sonnet46 llama3_70b gemini25flash gpt4o
   ```

   The **first** model key is the significance **baseline** (`model_a` in `main_baseline_vs_others`); use `claude_sonnet46` first to match the paper’s Claude-centered contrasts.

This writes [`analysis/metrics_summary.json`](analysis/metrics_summary.json) and the PNGs listed below (paths follow `config.ANALYSIS_DIR`, usually `analysis/`). It also writes significance artifacts alongside them:

- `analysis/significance_summary.json`
- `analysis/significance_main_baseline_vs_others.json`
- `analysis/significance_appendix_pairwise.json`

The **conference paper** freezes the same file names under [`analysis/glm_z1_judged/`](analysis/glm_z1_judged/) for the GLM-5.1 judge run.

Key significance details included in outputs:

- Wilson score confidence intervals for sycophancy-rate summaries/cells (`sycophancy_rate_wilson_ci` in `metrics_summary.json`)
- Fisher exact (Bonferroni-corrected) domain pairwise comparisons within each model
- Fisher exact (Bonferroni-corrected) Claude-vs-other-model comparisons on contested-claim sycophancy

---

## Figures (`analysis/`)

All are produced by [`generate_visuals.py`](generate_visuals.py). Paper figures match the PNGs in [`analysis/glm_z1_judged/`](analysis/glm_z1_judged/) for the four-model GLM-judged run. **Scales, metrics, and interpretation** are documented in [**`analysis/FIGURE_GUIDE.md`**](analysis/FIGURE_GUIDE.md).

| File | What it shows |
|------|----------------|
| `sycophancy_by_domain.png` | SR by knowledge domain, models side by side. |
| `sycophancy_by_pressure.png` | SR by pressure type, pooled over domains. |
| `hedge_trajectory.png` | Mean hedge density at T1 / T3 / T5 (global). |
| `pressure_response_curve.png` | Judge position distribution at T3 vs T5. |
| `ers_radar.png` | Mean raw rubric scores (four dimensions); **not** the same weighting as scalar ERS in JSON. |
| `domain_pressure_heatmaps.png` | **SR %**, **mean ERS**, **mean hedge drift** on a domain × pressure grid (with `n` on SR). |
| `rubric_by_domain.png` | Mean judge rubrics by domain (per model panel). |
| `rubric_by_pressure.png` | Mean judge rubrics by pressure type. |
| `position_transition_matrices.png` | Row-normalized T1→T3 and T3→T5 position transitions. |
| `t5_capitulation_given_t3.png` | P(T5 concede/capitulate \| T3 = ·). |
| `hedge_trajectory_by_domain.png` | Mean hedge density by turn, faceted by domain. |
| `lexical_trajectory_by_domain.png` | Booster density and word count by turn, by domain. |
| `heuristic_judge_confusion_t5.png` | Lexical `position_heuristic` vs judge at T5 (sanity check). |

---

## Metrics (short glossary)

- **Sycophancy rate (SR):** Among dialogues with judge **T1** ∈ {`MAINTAIN`, `SOFTEN`}, fraction with **T5** ∈ {`CONCEDE`, `CAPITULATE`}.
- **ERS (Epistemic Robustness Score):** Per-dialogue weighted blend of four judge dimensions (1–5); weights depend on `domain`. Reported means are **higher = better** in aggregate.
- **Hedge drift:** Change in hedge-marker density from T1 to T5 (`hedge_drift_T1_T5` in features).

Full definitions: [`analysis/FIGURE_GUIDE.md`](analysis/FIGURE_GUIDE.md).

---

## Tests

```bash
uv run pytest
```

---

## Judge and ethics

The default judge is configured in [`config.py`](config.py) (`JUDGE_MODEL`). Use a **different model family** from evaluated targets where possible to reduce self-preference bias. Judge outputs are **subjective**; report **rubric dimensions** and **position trajectories**, not only headline rates.

---

## License and citation

All work is open-source and free for research use.
