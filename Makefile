EXP ?=
SOURCE ?= templates/experiment
SURVEY_TITLE ?=
SURVEY_SLUG ?=
EXTRA_ARGS ?=
VALIDATE_ARGS ?=
SUBMISSION ?= submission.csv
SUBMISSION_REF ?=
STATUS ?=
CV ?=
PUBLIC_LB ?=
PRIVATE_LB ?=
METRIC ?=
KEY_IDEA ?=
NOTES ?=
NOTEBOOK ?= inference
KERNEL ?=
KERNEL_VERSION ?= 1
COMPETITION ?=
PROJECT_COMPETITION = $(shell PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/project_value.py competition.slug)
RESOLVED_COMPETITION = $(if $(strip $(COMPETITION)),$(COMPETITION),$(PROJECT_COMPETITION))
MESSAGE ?= $(EXP)
OUTPUT_FILE ?= submission.csv
OUT ?= /tmp/kaggle-output/$(EXP)/$(NOTEBOOK)
PROJECT_VIEWER_DATA = $(shell PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/project_value.py data.raw_dir)
VIEWER_DATA ?= $(PROJECT_VIEWER_DATA)
UV_CACHE_DIR ?= /tmp/uv-cache
PYTHONDONTWRITEBYTECODE ?= 1
export UV_CACHE_DIR
export PYTHONDONTWRITEBYTECODE

.PHONY: validate-template validate-config check-strategy-docs new-exp new-survey-report update-survey-index validate-surveys validate-exp check-exp check-skills check-skill-modules test-exp test-common train-local infer-local dl-kaggle-comp fetch-kaggle-notebooks archive-kaggle-discussions submit-check submit-code pipeline-local prepare-kaggle-notebooks push-kaggle-notebook push-kaggle-train push-kaggle-infer execute-notebook-local kaggle-status kaggle-logs kaggle-output record-submission record-exp compare-exp metric-weighted-tail-error-map pf-beam-disagreement-error-map update-summary app oof-app viewer viewer-smoke fmt test

validate-template:
	.venv/bin/python scripts/validate_project.py
	.venv/bin/python scripts/check_strategy_docs.py
	.venv/bin/python scripts/update_survey_index.py --check --allow-draft
	.venv/bin/python scripts/update_experiment_summary.py --check
	.venv/bin/python scripts/check_markdown_links.py

validate-config:
	.venv/bin/python scripts/validate_project.py --strict $(VALIDATE_ARGS)
	.venv/bin/python scripts/check_strategy_docs.py
	.venv/bin/python scripts/update_survey_index.py --check --allow-draft
	.venv/bin/python scripts/update_experiment_summary.py --check
	.venv/bin/python scripts/check_markdown_links.py

check-strategy-docs:
	.venv/bin/python scripts/check_strategy_docs.py

new-exp:
	.venv/bin/python scripts/new_experiment.py --name $(EXP) --source $(SOURCE) $(EXTRA_ARGS)

new-survey-report:
	.venv/bin/python scripts/new_survey_report.py --title "$(SURVEY_TITLE)" --slug "$(SURVEY_SLUG)" $(EXTRA_ARGS)

update-survey-index:
	.venv/bin/python scripts/update_survey_index.py

validate-surveys:
	.venv/bin/python scripts/update_survey_index.py --check

validate-exp:
	.venv/bin/python scripts/validate_experiment.py --experiment $(EXP) $(EXTRA_ARGS)

check-exp:
	.venv/bin/ruff check experiments/$(EXP)
	.venv/bin/ruff format --check experiments/$(EXP)

check-skills:
	.venv/bin/python scripts/validate_skills.py
	uv run --extra dev ruff check .agents/skills

# Compatibility alias for the former, narrower target name.
check-skill-modules: check-skills

test-exp:
	@if [ -d experiments/$(EXP)/tests ]; then \
		uv run --extra dev --extra notebook pytest -q experiments/$(EXP)/tests; \
	else \
		echo "No experiment-specific tests: experiments/$(EXP)/tests"; \
	fi

test-common:
	uv run --extra dev --extra notebook pytest -q tests

# Debug-only. Kaggle notebook execution is authoritative; pass
# EXTRA_ARGS="--allow-local ..." to opt in to local smoke execution.
train-local:
	.venv/bin/python scripts/execute_experiment_notebook.py --experiment $(EXP) --notebook train $(EXTRA_ARGS)

# Debug-only. Kaggle notebook execution is authoritative; pass
# EXTRA_ARGS="--allow-local ..." to opt in to local smoke execution.
infer-local:
	.venv/bin/python scripts/execute_experiment_notebook.py --experiment $(EXP) --notebook inference $(EXTRA_ARGS)

dl-kaggle-comp:
	.venv/bin/python scripts/kaggle_download.py

fetch-kaggle-notebooks:
	.venv/bin/python .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition $(RESOLVED_COMPETITION) $(EXTRA_ARGS)

archive-kaggle-discussions:
	.venv/bin/python scripts/archive_kaggle_discussions.py --competition $(RESOLVED_COMPETITION) $(EXTRA_ARGS)

submit-check:
	.venv/bin/python scripts/validate_submission.py --submission "$(SUBMISSION)" --experiment "$(EXP)"

submit-code:
	.venv/bin/kaggle competitions submit $(RESOLVED_COMPETITION) -k $(KERNEL) -v $(KERNEL_VERSION) -f $(OUTPUT_FILE) -m "$(MESSAGE)"

# Debug-only local pipeline; Kaggle kernels remain the source of truth.
pipeline-local:
	$(MAKE) validate-exp EXP=$(EXP) EXTRA_ARGS="$(VALIDATE_ARGS)"
	$(MAKE) train-local EXP=$(EXP) EXTRA_ARGS="$(EXTRA_ARGS)"
	$(MAKE) infer-local EXP=$(EXP) EXTRA_ARGS="$(EXTRA_ARGS)"
	$(MAKE) submit-check EXP=$(EXP) SUBMISSION=$(SUBMISSION)

prepare-kaggle-notebooks:
	.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment $(EXP) --strict $(EXTRA_ARGS)

push-kaggle-notebook:
	.venv/bin/python scripts/validate_kaggle_metadata.py --package-dir experiments/$(EXP)/kaggle/$(NOTEBOOK)
	.venv/bin/kaggle kernels push -p experiments/$(EXP)/kaggle/$(NOTEBOOK)

push-kaggle-train:
	$(MAKE) push-kaggle-notebook EXP=$(EXP) NOTEBOOK=train

push-kaggle-infer:
	$(MAKE) push-kaggle-notebook EXP=$(EXP) NOTEBOOK=inference

execute-notebook-local:
	.venv/bin/python scripts/execute_experiment_notebook.py --experiment $(EXP) --notebook $(NOTEBOOK) $(EXTRA_ARGS)

kaggle-status:
	.venv/bin/kaggle kernels status $(KERNEL)

kaggle-logs:
	.venv/bin/kaggle kernels logs -f $(KERNEL)

kaggle-output:
	mkdir -p $(OUT)
	.venv/bin/kaggle kernels output $(KERNEL) -p $(OUT)

record-submission:
	.venv/bin/python scripts/record_submission.py --experiment $(EXP) --file $(SUBMISSION) --submission-ref "$(SUBMISSION_REF)" $(EXTRA_ARGS)

record-exp:
	.venv/bin/python scripts/record_experiment.py --experiment $(EXP) --status "$(STATUS)" --cv "$(CV)" --public-lb "$(PUBLIC_LB)" --private-lb "$(PRIVATE_LB)" --metric "$(METRIC)" --key-idea "$(KEY_IDEA)" --notes "$(NOTES)" $(EXTRA_ARGS)

compare-exp:
	.venv/bin/python scripts/compare_experiments.py $(EXTRA_ARGS)

metric-weighted-tail-error-map:
	.venv/bin/python scripts/metric_weighted_tail_error_map.py $(EXTRA_ARGS)

pf-beam-disagreement-error-map:
	.venv/bin/python scripts/pf_beam_disagreement_error_map.py $(EXTRA_ARGS)

update-summary:
	.venv/bin/python scripts/update_experiment_summary.py

app:
	.venv/bin/streamlit run app/streamlit_app.py

oof-app:
	.venv/bin/streamlit run app/oof_analysis_app.py

viewer:
	UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --extra viewer python scripts/run_rogii_viewer.py --dataset "$(VIEWER_DATA)" $(EXTRA_ARGS)

viewer-smoke:
	UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --extra viewer python scripts/run_rogii_viewer.py --dataset "$(VIEWER_DATA)" --smoke-test $(EXTRA_ARGS)

fmt:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

test:
	uv run --extra dev --extra notebook pytest
