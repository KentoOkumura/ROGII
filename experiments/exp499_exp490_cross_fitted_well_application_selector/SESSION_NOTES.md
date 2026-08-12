# exp499_exp490_cross_fitted_well_application_selector セッションノート

## 目的

exp490を適用すべきwellと保存済みexp357へ戻すべきwellを、suffix truthより前の情報だけで
見極められるか、strict-nested OOFで調査する。

## 現在の状態

- Route: `ensemble`
- 状態: `completed_fail_closed`
- CV: cross-fitted policy `8.514310626`、always-exp490 `8.480155260`
- LB: なし
- inference/submission: 無効

## 事前根拠

- exp490 always: RMSE 8.480155260、exp357 9.737195157より1.25704 ft改善。
- by-well: 449 wells改善、324 wells悪化。
- report-only oracle well routing: RMSE 6.560582422。always-exp490比1.91957 ftの余地。
- strongest exploratory target-free signal: mean absolute exp357-exp226 disagreement、
  pooled beneficial-well AUC 0.5919、fold AUC 0.5644–0.6422、正方向5/5。
- この探索値は最終判定ではなく、別expを作る根拠に限定する。

## 固定設計

- target: `exp357_rmse^2 - exp490_rmse^2`、row数でweight。
- target-free 32特徴、outer 5 / inner 4、threshold 0固定。
- inner policy候補: always-exp490、weighted Ridge、weighted shallow HGB。
- outer-validを見ずにmodel familyを選び、held-out wellへ一度だけ適用する。
- predictabilityとsafe-routerを別gateで判定し、両方通らない限りinferenceへ進めない。

## Push前の実行量確認

- variants: 1
- learned model configs: 2
- outer folds: 5
- inner folds per outer: 4
- inner fits: 40
- maximum outer refits: 5
- maximum total CPU model fits: 45
- LightGBM configs / boosters: 0 / 0
- new PF / HMM / Beam / candidate predictions: 0 / 0 / 0 / 0
- parent/control retraining: 0
- GPU runs: 0

既存controlの再学習を含まず、保存済みOOFだけを使うため追加GPU承認は不要。

## 再現性メモ

- seed policy: fixed 42、HGB `random_state=42`
- stochastic components: sklearn HGBのみ
- parallel RNG: single process、joblib並列なし
- CPU/GPU: Kaggle CPU / GPU 0
- upstream prediction gzip raw SHA: `99030b33...61b72c`
- upstream prediction decompressed SHA: `e020e82e...e9a07`
- upstream by-well SHA: `65abf013...ba076`
- exp498 feature SHA: `c1d31113...0f5ad`
- exp499 feature SHA: `54c7e1da...7bb0d4`
- selector OOF SHA: `8b9a44d...43d6610`
- model manifest SHA: `72577e6d...a55ac1`
- submission SHA: not applicable

## コマンドログ

### 2026-08-01

```bash
make new-steering EXP=exp499_exp490_cross_fitted_well_application_selector
make new-exp EXP=exp499_exp490_cross_fitted_well_application_selector
.venv/bin/pytest -q experiments/exp499_exp490_cross_fitted_well_application_selector/tests/test_exp499_contract.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp499_exp490_cross_fitted_well_application_selector/exp499_exp490_cross_fitted_well_application_selector_compact_selfcontained_train.py
```

- contract tests: 4/4 PASS
- syntax / `ruff --select F821` / both estimator fit-predict smoke: PASS
- compact train 21 cells、inference guard 3 cellsを確認し、template placeholderより正規notebookへ採用した。

### Kaggle初回実行

```bash
make validate-exp EXP=exp499_exp490_cross_fitted_well_application_selector
make prepare-kaggle-notebooks EXP=exp499_exp490_cross_fitted_well_application_selector \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp499-exp490-cross-fitted-well-selector-train --title 'exp499 exp490 cross fitted well selector train' --run-on-push --strict"
make push-kaggle-train EXP=exp499_exp490_cross_fitted_well_application_selector
```

- validate-exp strict: PASS
- Kaggle package contract tests: 4/4 PASS
- metadata: private / CPU / internet off / run-on-push
- sources: `exp490-mean-revert-full-merge`、`exp498-geometry-mean-reversion-tail-regime-train`
- canonical kernel: `kentookumura/exp499-exp490-cross-fitted-well-selector-train`
- version 1: `ERROR`。exp498 feature contract file SHAがローカル仮実行版
  `23540b...`とKaggle正本`ccdc3d...`で異なった。feature content SHA
  `c1d311...`は一致し、差分は`well` dtype表記`str` / `object`と、それに伴う
  logical contract SHAだけだった。
- 修正: Kaggle正本のfile SHA `ccdc3d...`、logical SHA `92d1e7...`へpinし、
  ローカル候補もdownload済み`kaggle/output/train_v2/artifacts`を優先する。
  特徴、model、fold、threshold、gate、実行量は変更しない。
- version 2: 同じcanonical kernel IDへpush成功、CPU再実行開始。

### Kaggle version 2 結果

- id_no / version: `129362815` / `2`
- status: `COMPLETE`
- runtime / peak RSS: `57.051717 sec` / `1.454178 GiB`
- technical checks: 9/9 PASS
- feature freeze: 773 wells / 32 features / SHA `54c7e1da...7bb0d4`
- actual execution: 45/45 CPU model fits、PF/HMM/Beam/GPU/control retraining 0
- predictability: pooled AUC `0.521151`、AUC>=0.55は1/5 folds、
  Spearman `0.122250`・正方向5/5。gate FAIL。
- safe router: RMSE `8.514310626`、always-exp490 `8.480155260`比
  `+0.034155367 ft`悪化。適用716/773 wells、beneficial precision `58.1006%`。
- tail: selected-minus-exp357 p95 / worst `+7.098191 / +49.602560 ft`。
  catastrophic 51 wells中48へexp490を適用。gate FAIL。
- fold policy: fold 0/2/3/4はalways-exp490、fold 1だけHGBを採用。
  fold 1は`8.659383 -> 8.822361 ft`へ悪化した。
- strongest univariate: `parent_exp226_abs_mean` AUC `0.591912`、fold min
  `0.564386`、Spearman正方向5/5。弱いranking signalは残るがrouterへ変換できない。
- verdict: `deployment_eligible=false`、inference/submissionなし、same-OOF rescueなし。

### Artifact回収

- output: `kaggle/output/train_v2`（2.9 MiB）
- feature table SHA: `54c7e1dac064f929edd57dc03bf00d1a15b47340d5c40d7e6e8afc3e707bb0d4`
- selector OOF SHA: `8b9a44d3bfd4b62203c2ac85598bb3c5970914cf769ad53f93e50041343d6610`
- fold metrics SHA: `f6e4cc7351f60272c70c9ab8b7832f6d81f63947b56c65a508f034e16ffe538a`
- model manifest SHA: `72577e6da61cfef4ec67679d1be1b635e645e652ec89fa37a03e1500bda55ac1`
- summary SHA: `d1ce774dff2fe113490616318291538e120714aa9a98fe2dccff34cfc0054413`
- metrics SHA: `05be2063b36940585e1aa4b73a6077ce3af923df46291a75b89550fc39ddf824`

## 次のアクション

1. 本selector branchをterminal closeする。
2. exp490のfail-closeを維持し、inference / submissionへ進めない。
3. exp500は別仮説・別承認のまま維持し、exp499 signalをadaptive gateへ使わない。
