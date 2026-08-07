# exp091_self_gr_likelihood_pf_beam_probe セッションノート

## 現在の状態

- status: `completed_train_side_audit`
- route: `pf_beam`
- parent: `exp090_lateral_self_gr_match_pseudotail_probe`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- blocked: none

## 実装内容

- `.steering/20260620-exp091-self-gr-likelihood-pf-beam-probe/` を作成。
- `experiments/exp091_self_gr_likelihood_pf_beam_probe/` を exp090 から作成。
- `settings.py` の experiment name を exp091 に更新。
- `config.yaml` を `all_horizontal_self_similarity_candidate_rank_audit` 用に更新。
- 補助実装を `self_gr_likelihood_pf_beam_probe.py` に差し替え。
  - exp072 deterministic train cache から `last_known_tvt`、`target`、`pf_ancc`、`beam_mean_d`、`likpf_mean_d`、`sc_ens_d`、`hyb_d`、PF/Beam confidence diagnostic を読む。
  - raw train horizontal well の `GR` と finite prefix `TVT_input` だけから self-GR 候補 `self_gr_ens` / `self_gr_best` / scale 別候補を作る。
  - `true_tvt = last_known_tvt + target` は coverage、oracle headroom、bucket miss rate の評価専用にする。
  - 候補別 metrics、oracle topK、target-free `candidate_rank_score` topK、distance/tail bucket、by-well metrics、candidate long frame を保存する。
- train notebook を exp091 用の4セクション構成に更新。
- inference notebook は diagnostic-only guard として停止する。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp091_self_gr_likelihood_pf_beam_probe
uv run python scripts/new_experiment.py --name exp091_self_gr_likelihood_pf_beam_probe --source experiments/exp090_lateral_self_gr_match_pseudotail_probe
```

## 次のアクション

1. 静的検証を通す。
2. synthetic smoke test で self-GR candidate / coverage summary の最低限の挙動を確認する。
3. Kaggle train package を作成し、bootstrap manifest の config / 補助 `.py` SHA を確認する。
4. Kaggle train を実行して、candidate coverage、rank headroom、bucket miss rate を読む。

## 検証

- `uv run python -m py_compile experiments/exp091_self_gr_likelihood_pf_beam_probe/self_gr_likelihood_pf_beam_probe.py experiments/exp091_self_gr_likelihood_pf_beam_probe/public_notebook_replay_audit.py experiments/exp091_self_gr_likelihood_pf_beam_probe/settings.py`: PASS
- `python3 -m json.tool experiments/exp091_self_gr_likelihood_pf_beam_probe/exp091_self_gr_likelihood_pf_beam_probe_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp091_self_gr_likelihood_pf_beam_probe/exp091_self_gr_likelihood_pf_beam_probe_inference.ipynb`: PASS
- `uv run ruff check experiments/exp091_self_gr_likelihood_pf_beam_probe/self_gr_likelihood_pf_beam_probe.py experiments/exp091_self_gr_likelihood_pf_beam_probe/public_notebook_replay_audit.py experiments/exp091_self_gr_likelihood_pf_beam_probe/settings.py`: PASS
- `uv run ruff format --check experiments/exp091_self_gr_likelihood_pf_beam_probe/self_gr_likelihood_pf_beam_probe.py experiments/exp091_self_gr_likelihood_pf_beam_probe/settings.py`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp091_self_gr_likelihood_pf_beam_probe`: PASS
- synthetic frame による `run_self_gr_likelihood_pf_beam_probe()` smoke test: PASS、24 rows / 2 wells、8 expected output files generated.
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp091_self_gr_likelihood_pf_beam_probe --notebook train --kernel-id kentookumura/exp091-self-gr-likelihood-pf-beam-probe-train --title "exp091 self gr likelihood pf beam probe train" --run-on-push --strict`: PASS
- generated train package: `experiments/exp091_self_gr_likelihood_pf_beam_probe/kaggle/train`
- generated kernel id: `kentookumura/exp091-self-gr-likelihood-pf-beam-probe-train`
- generated metadata: GPU disabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated bootstrap manifest includes `config.yaml` SHA `56a7fb34bf39e6caea469dfbd8ae19790eeddae572aeb7c1629e5b5793bc8783` and `self_gr_likelihood_pf_beam_probe.py` SHA `4d18e4965d48178f1702d097568e3a88ee05777bbab2dd681dfbe7e1ed2668fe`。

## Kaggle train v1

```bash
make push-kaggle-train EXP=exp091_self_gr_likelihood_pf_beam_probe
kaggle kernels status kentookumura/exp091-self-gr-likelihood-pf-beam-probe-train
kaggle kernels logs kentookumura/exp091-self-gr-likelihood-pf-beam-probe-train
kaggle kernels output kentookumura/exp091-self-gr-likelihood-pf-beam-probe-train -p experiments/exp091_self_gr_likelihood_pf_beam_probe/kaggle/output/train
```

- kernel: `kentookumura/exp091-self-gr-likelihood-pf-beam-probe-train`
- version: 1
- status: `COMPLETE`
- runtime: 1,533.571 sec
- rows: 3,783,989
- wells: 773
- source cache: `/kaggle/input/notebooks/kentookumura/exp072-exp063-full-replay-feature-cache-train/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- source SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- output: `experiments/exp091_self_gr_likelihood_pf_beam_probe/kaggle/output/train/artifacts/`

### 主要結果

- best single candidate: `likpf_mean`, RMSE 11.594897, MAE 7.067633, within 10ft 0.772807.
- `pf_ancc`: RMSE 14.493051, within 10ft 0.691741.
- `beam_mean`: RMSE 15.774327, within 10ft 0.591649.
- `self_gr_ens`: RMSE 191.215912, within 10ft 0.135627.
- `self_gr_best`: RMSE 250.161697, within 10ft 0.270874.
- oracle best candidate: RMSE 6.873199, within 10ft 0.925153, selected self-GR rate 0.135212.
- target-free `candidate_rank_score` top1: RMSE 29.985529, within 10ft 0.746819.
- target-free `candidate_rank_score` top10: RMSE 6.953187, within 10ft 0.922684.

### 判断

self-GR candidate 単体は弱く、直接置換や hard switch は不採用。oracle headroom は大きいが、現行 target-free rank score は top1 selector として不十分。後続に進む場合は `pf_candidate_coverage_then_ranker_audit` / supervised ranker の補助特徴として self-GR score を使う範囲に限定する。
