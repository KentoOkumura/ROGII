EXP ?= exp001_baseline
SOURCE ?= templates/experiment
SURVEY_TITLE ?=
SURVEY_SLUG ?=
EXTRA_ARGS ?=
VALIDATE_ARGS ?=
SUBMISSION ?= submission.csv
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
COMPETITION ?= rogii-wellbore-geology-prediction
MESSAGE ?= $(EXP)
OUTPUT_FILE ?= submission.csv
OUT ?= /tmp/kaggle-output/$(EXP)/$(NOTEBOOK)
VIEWER_DATA ?= data/raw
UV_CACHE_DIR ?= /tmp/uv-cache

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: validate-template validate-config new-exp new-steering new-survey-report update-survey-index validate-surveys validate-exp train-local infer-local dl-kaggle-comp submit-check submit-code pipeline-local prepare-kaggle-notebooks push-kaggle-train push-kaggle-infer execute-notebook-local kaggle-status kaggle-logs kaggle-output record-submission record-exp compare-exp update-summary app oof-app viewer viewer-smoke fmt test

validate-template:
	.venv/bin/python scripts/validate_project.py
	.venv/bin/python scripts/update_survey_index.py --check

validate-config:
	.venv/bin/python scripts/validate_project.py --strict
	.venv/bin/python scripts/update_survey_index.py --check

new-exp:
	.venv/bin/python scripts/new_experiment.py --name $(EXP) --source $(SOURCE) $(EXTRA_ARGS)

new-steering:
	.venv/bin/python scripts/new_steering.py --experiment $(EXP) $(EXTRA_ARGS)

new-survey-report:
	.venv/bin/python scripts/new_survey_report.py --title "$(SURVEY_TITLE)" --slug "$(SURVEY_SLUG)" $(EXTRA_ARGS)

update-survey-index:
	.venv/bin/python scripts/update_survey_index.py

validate-surveys:
	.venv/bin/python scripts/update_survey_index.py --check

validate-exp:
	.venv/bin/python scripts/validate_experiment.py --experiment $(EXP) $(EXTRA_ARGS)

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

submit-check:
	.venv/bin/python scripts/validate_submission.py --submission $(SUBMISSION)

submit-code:
	kaggle competitions submit $(COMPETITION) -k $(KERNEL) -v $(KERNEL_VERSION) -f $(OUTPUT_FILE) -m "$(MESSAGE)"

# Debug-only local pipeline; Kaggle kernels remain the source of truth.
pipeline-local:
	$(MAKE) validate-exp EXP=$(EXP) EXTRA_ARGS="$(VALIDATE_ARGS)"
	$(MAKE) train-local EXP=$(EXP) EXTRA_ARGS="$(EXTRA_ARGS)"
	$(MAKE) infer-local EXP=$(EXP) EXTRA_ARGS="$(EXTRA_ARGS)"
	$(MAKE) submit-check EXP=$(EXP) SUBMISSION=$(SUBMISSION)
	$(MAKE) update-summary

prepare-kaggle-notebooks:
	.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment $(EXP) $(EXTRA_ARGS)

push-kaggle-train:
	kaggle kernels push -p experiments/$(EXP)/kaggle/train

push-kaggle-infer:
	kaggle kernels push -p experiments/$(EXP)/kaggle/inference

execute-notebook-local:
	.venv/bin/python scripts/execute_experiment_notebook.py --experiment $(EXP) --notebook $(NOTEBOOK) $(EXTRA_ARGS)

kaggle-status:
	kaggle kernels status $(KERNEL)

kaggle-logs:
	kaggle kernels logs -f $(KERNEL)

kaggle-output:
	mkdir -p $(OUT)
	kaggle kernels output $(KERNEL) -p $(OUT)

record-submission:
	.venv/bin/python scripts/record_submission.py --experiment $(EXP) --file $(SUBMISSION) $(EXTRA_ARGS)

record-exp:
	.venv/bin/python scripts/record_experiment.py --experiment $(EXP) --status "$(STATUS)" --cv "$(CV)" --public-lb "$(PUBLIC_LB)" --private-lb "$(PRIVATE_LB)" --metric "$(METRIC)" --key-idea "$(KEY_IDEA)" --notes "$(NOTES)" $(EXTRA_ARGS)

compare-exp:
	.venv/bin/python scripts/compare_experiments.py $(EXTRA_ARGS)

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
	.venv/bin/pytest
