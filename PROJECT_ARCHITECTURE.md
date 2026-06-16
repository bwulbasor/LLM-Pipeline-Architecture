# Project Architecture

## Goal

Build a reproducible pipeline for evaluating recommender-system explanations with LLM-generated personas.

## High-Level Flow

```text
MovieLens 10M
  -> preprocessing + filtering
  -> user feature engineering
  -> archetype clustering
  -> seed user selection
  -> recommendation/explanation task generation
  -> persona prompt construction
  -> balanced evaluation batch construction
  -> local simulated evaluation OR tiny live evaluation
  -> metrics + figures
  -> ablation experiments
```

## Main Components

### 1. Orchestration

`run_pipeline.py`

- supports stage skipping when outputs are fresh
- supports partial reruns with `--start-at` / `--end-at`
- supports `local` and `local_with_live_small` modes
- writes `results/pipeline_run_summary.json`

### 2. Configuration

`src/config.py`

- filtering thresholds
- clustering parameters
- evaluation defaults
- ablation defaults
- live backend defaults

### 3. Preprocessing and Archetypes

`src/step1_preprocess.py`

- downloads/loads MovieLens
- filters users/items
- applies a model knowledge cutoff
- engineers user behavior features
- clusters users with KMeans
- labels five archetypes
- selects representative seed users

Outputs:

- `data/user_features.csv`
- `data/filtered_ratings.csv`
- `data/movies.csv`
- `results/seed_users.json`
- `results/archetype_profiles.json`
- `results/clustering_diagnostics.json`

### 4. Recommendation and Explanation Task Generation

`src/step2_generate_explanations.py`

- builds nearest-neighbor lookup
- summarizes user histories
- scores candidate movies with a lightweight hybrid recommender signal using:
  - content similarity to top-rated anchors
  - genre affinity from rating deviations
  - neighbor support and average neighbor rating
  - popularity bonus
- generates three explanation families:
  - feature
  - neighbor
  - counterfactual
- creates adversarial flawed explanations

Outputs:

- `results/evaluation_tasks.json`
- `results/seed_user_histories.json`
- `results/adversarial_explanations.json`

### 5. Persona Prompting

`src/step3_personas.py`

- Strategy A: trait-only personas
- Strategy B: demographic-augmented personas
- persona granularities:
  - thin
  - standard
  - thick
- calibration prompt variants
- moderator prompt template

Output:

- `results/persona_templates.json`

### 6. Batch Construction

`src/step4_batch_eval.py`

- stratified balanced sampling by archetype and explanation type
- constructs:
  - compact batch
  - full batch
  - adversarial batch

Outputs:

- `results/eval_batch_compact.json`
- `results/eval_batch_full.json`
- `results/eval_batch_adversarial.json`

### 7. Local Simulated Evaluation

`src/step4_generate_evaluations.py`

- simulates persona-based scoring
- supports:
  - strategy A/B
  - thin/standard/thick
  - calibration levels
  - panel size
  - debate rounds
  - history length
  - temperature
  - random seed
- computes built-in experiment blocks for:
  - RQ1 strategy comparison
  - RQ2 single vs multi-agent
  - RQ3 proxy cognitive-load comparison
- generates baselines:
  - zero-shot
  - single-agent
  - heuristic
  - misaligned persona

Outputs:

- `results/experiment_results.json`
- `results/baseline_results.json`

### 8. Optional Live Evaluation

`src/step4_multi_agent_eval.py`

- OpenAI-compatible live evaluation path
- NVIDIA endpoint is the default backend
- intended for tiny live validation samples, not heavy full-scale execution

Outputs:

- `results/experiment_results_live.json`
- or `results/experiment_results_live_small.json`

### 9. Analysis and Metrics

`src/step5_analysis.py`

Computes:

- Human Alignment Score (HAS)
- Spearman rank alignment
- Krippendorff alpha
- ICC
- disconfirmation score
- cognitive-load differential
- baseline comparison
- implementation audit

Outputs:

- `results/metrics_report.json`
- plot files under `figures/`

### 10. Ablations

`src/step6_ablations.py`

Runs scaled-down ablations over:

- persona granularity
- debate rounds
- calibration
- panel size
- persona strategy
- temperature
- history length
- random seed

Outputs:

- `results/ablation_results.json`
- `results/ablation_summary.json`

## Current Scope

This is a student-scale research implementation.

It is intentionally:

- reproducible
- transparent
- scaled down where necessary

It is not intended to be:

- a production recommender service
- a large-scale multi-provider evaluation platform
- a fully human-validated benchmark suite

## Important Limitations

- LiDES is used as a silver-standard benchmark, not a perfect direct human-label match
- the live free-model path is practical for tiny samples, not large multi-agent sweeps
- the cognitive-load experiment is a proxy operationalization
