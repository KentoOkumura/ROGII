# exp308_imputed_gr_confidence_downweight セッションノート

## 目的

exp307補間GRを変えず、元のmissing行だけGR evidenceを距離依存で弱める。

## 現在の状態

- Route: `pf_beam`
- 状態: exp307 promotion gate FAILにより未実行のまま閉鎖
- variants / HMM runs / boosters: `1 / 773 / 0`
- parent再実行: 0

## 2026-07-21 設計

```bash
make new-steering EXP=exp308_imputed_gr_confidence_downweight
make new-exp EXP=exp308_imputed_gr_confidence_downweight
```

- weight式、dependency、gap readout、promotion gate、禁止gridを固定した。
- 実装、Kaggle package/push/run、inference、submissionは行っていない。

## 2026-07-21 実装

- ユーザーの「exp308を実装してください」を実装のみの承認として記録した。Kaggle package/push/run承認は含めていない。
- `exp308_imputed_gr_confidence_downweight_compact_selfcontained_train.py`をJupytext percent形式で実装し、compact/正規train Notebookを生成した。
- exp307 v1が実行中のためdependency status、prediction/scale/input/promotion SHA、parent direct/blend metricsは`PENDING_EXP307_PASS`のままにした。`parent_dependency_frozen=false`ではHMM開始前に停止する。
- raw GRから同一well内の最近傍finite raw GRまでの行距離をO(n)で計算し、finite行exact 1、missing行`max(0.25,2^(-d/8))`、全GR non-finite時0.25を実装した。
- exp307のevaluation GR interpolationを同じpandas linear interpolation + both-direction + typewell mean fallbackで維持し、exp307保存finite-MAD scaleをwell別に読む契約にした。
- exp209 exact forward-backward kernelをself-containedに維持し、Gaussian log emissionへrow weightを1回だけ乗算した。grid、transition、prior、posterior meanは変更していない。
- mask/distance/weight/interpolated GR auditとcandidate predictionをgzip content SHA付きでfreezeした後にだけtruth/folds/hidden-like/LikPFを読むlate joinにした。
- overall/fold/1000+/hidden-like/by-wellに加え、observed/missing、gap 1--3 / 4--15 / 16+、exact distance readoutとfixed LikPF 50:50 guardを実装した。
- `exp308_imputed_gr_confidence_downweight_compact_selfcontained_inference.py`と正規inference Notebookはraw-test prediction/submissionを明示的にfail-closeした。
- 親compactとの比較はexp307 `1,676行/10章`、exp308 `1,987行/10章`で、同一exp helper importだけの薄いNotebookではない。

### 実行量ガード

- active variants: 1 (`missing_distance_half8_floor025`)
- HMM well-runs: `1 x 773 = 773`
- model / LightGBM configs / trained folds / PF / Beam / boosters: `0 / 0 / 0 / 0 / 0 / 0`
- parent/control再実行: 0
- Kaggle GPU: 0、CPU予定、internet off

### 検証コマンド

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp308_imputed_gr_confidence_downweight/exp308_imputed_gr_confidence_downweight_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp308_imputed_gr_confidence_downweight/exp308_imputed_gr_confidence_downweight_compact_selfcontained_inference.py
.venv/bin/python -m py_compile experiments/exp308_imputed_gr_confidence_downweight/exp308_imputed_gr_confidence_downweight_compact_selfcontained_train.py experiments/exp308_imputed_gr_confidence_downweight/exp308_imputed_gr_confidence_downweight_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp308_imputed_gr_confidence_downweight/exp308_imputed_gr_confidence_downweight_compact_selfcontained_train.py experiments/exp308_imputed_gr_confidence_downweight/exp308_imputed_gr_confidence_downweight_compact_selfcontained_inference.py experiments/exp308_imputed_gr_confidence_downweight/tests/test_exp308_imputed_gr_confidence_downweight.py --select F821,F811,F601
.venv/bin/pytest -q experiments/exp308_imputed_gr_confidence_downweight/tests/test_exp308_imputed_gr_confidence_downweight.py
make validate-exp EXP=exp308_imputed_gr_confidence_downweight
make validate-template
```

- exp308 contract tests: `12 passed`
- 最初のtestでtrailing gapの片側sentinelが短く距離を過小評価する不具合を検出し、`2*n+1`へ修正後にleading/trailing/no-finite契約をPASSした。
- 構文、未定義/重複定義、Jupytext round-trip: PASS。
- Notebook実行、Kaggle package/push/run、output取得、inference、submissionは未実施。

## 再現性メモ

- RNGなし。raw missing maskからdistance/weightを決定的に生成する。
- exp307 scientific contract/input/prediction SHAはparent gate FAILのためexp308 dependencyとして固定しない。
- `PENDING_EXP307_PASS` sentinelと`parent_dependency_frozen=false`を保持し、誤実行をfail-closeする。
- exp308を実行していないためmask/distance/weight/prediction/metrics content SHAは生成していない。

## 次のアクション

なし。exp307 gate FAILにより固定dependencyは成立しない。別parentへの再設計は独立した根拠と事前設計、ユーザー確認がある場合だけ扱う。

## 2026-07-22 dependency close

exp307 v2は正常完走したが、finite MAD primaryがdirect `+3.723054 ft`、fixed blend `+0.917640 ft`悪化し、全promotion gateをFAILした。事前手順を適用し、exp308はpackage/push/run、inference、submissionを行わず閉じる。実装済みNotebookとtestsは履歴として保持する。
