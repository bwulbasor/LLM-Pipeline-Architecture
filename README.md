# LLM Bias Pipeline Architecture

A student-scale research pipeline for evaluating recommender-system explanations with LLM-generated personas.

## What This Project Does

This project implements a full experimental workflow for:

- preprocessing MovieLens 10M
- discovering data-driven user archetypes
- generating recommendation/explanation tasks
- simulating persona-based single-agent and multi-agent evaluation
- auditing adversarial or flawed explanations
- running ablations
- optionally validating a tiny live sample with an LLM API

The project is designed as a reproducible assignment deliverable rather than a production recommender system.

## Main Pipeline

The local pipeline runs these stages:

1. preprocessing and archetype clustering
2. grounded task generation
3. persona prompt export
4. stratified evaluation batch construction
5. local simulated evaluation and baselines
6. metrics and figure generation
7. scaled-down ablations

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for the detailed structure.

## Key Files

- `run_pipeline.py` - orchestrates the project pipeline
- `src/config.py` - central configuration
- `src/step1_preprocess.py` - data filtering, feature engineering, clustering
- `src/step2_generate_explanations.py` - lightweight hybrid recommendation scoring and explanation synthesis
- `src/step3_personas.py` - persona strategies and prompt templates
- `src/step4_batch_eval.py` - balanced evaluation batch generation
- `src/step4_generate_evaluations.py` - local simulated evaluations
- `src/step4_multi_agent_eval.py` - optional live evaluation path
- `src/step5_analysis.py` - metrics and plots
- `src/step6_ablations.py` - ablations and sensitivity checks

## Running The Local Pipeline

```powershell
python run_pipeline.py
```

Resume later stages only:

```powershell
python run_pipeline.py --start-at 4 --end-at 7
```

Force reruns:

```powershell
python run_pipeline.py --force
```

## Tiny Live Sample

The project also supports a small live validation run using an OpenAI-compatible endpoint such as NVIDIA's hosted models.

Required environment variables:

- `OPENAI_COMPAT_API_KEY`
- optional: `OPENAI_COMPAT_BASE_URL`
- optional: `LLM_MODEL`

Example:

```powershell
python run_pipeline.py --mode local_with_live_small --live-sample-size 1 --live-panel-size 1 --live-debate-rounds 0 --live-reasoning false
```

## Repository Hygiene

This repository excludes:

- raw MovieLens downloads
- generated results
- generated figures
- local notes
- environment files and secrets

That keeps the public repo lightweight and prevents accidental key exposure.
