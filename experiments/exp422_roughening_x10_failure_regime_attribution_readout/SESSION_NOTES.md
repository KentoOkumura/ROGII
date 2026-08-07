# exp422_roughening_x10_failure_regime_attribution_readout セッションノート

## 目的

exp416でpersistent-offset SSEだけが24.700364%改善した一方、overall、5 folds、
全stress scope、well tailが悪化した理由を、保存診断によるtarget-free well regimeへ
帰属する。

## 変更点

exp416のPFやcontrolを再実行せず、固定manifestに属する保存生成物だけを使う。
outcome前に2軸regime / row scopeをfreezeし、その後だけgainとepisode outcomeを結合する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU audit version 2完了・scientific FAIL・branch close
- CV / LB: attribution readoutのためなし
- 実装承認: あり（2026-07-28ユーザー依頼）
- 正規train Notebook採用 / package / push / audit run承認: あり
- 実行承認: あり（2026-07-28ユーザー依頼）
- inference / submission: 禁止

## 2026-07-28 設計確定

- latest experimentがexp421であることを確認し、exp422を採番した。
- steeringを先に作り、その後design-only scaffoldを作成した。
- 親をexp416、controlをexp072、reporting sourceをexp226に固定した。
- exp416 merge version 2 / id_no `128912230` / artifact manifest SHA
  `708bb257e3ab360f09821823d5413fa9e1c5c32ef4ddb4917b41573943dffb86`
  をsource rootに固定した。
- 6 raw diagnostics、2 equal-weight scores、outer-4-fold経験分布順位、
  fold-safe median、1 target cellをoutcome前に固定した。
- scientific gateを2方向相関、4/5 fold、fixed cell gain、episode supportのANDにした。
- `docs/06_reproducibility.md`に従い、truth-late freeze、raw / decompressed /
  logical SHA、fold別SHA256 permutation seedを固定した。

## 2026-07-28 compact実装

- ユーザーの`exp422を実装してください`を実装承認として記録した。
- 正規train / inference Notebookは上書きせずplaceholderのまま保持した。
- 次のJupytext percent形式compact self-contained候補を作成し、`.ipynb`へ変換した。
  - `exp422_roughening_x10_failure_regime_attribution_readout_compact_selfcontained_train.py`
  - `exp422_roughening_x10_failure_regime_attribution_readout_compact_selfcontained_train.ipynb`
- 実装した順序:
  1. exp416 manifest raw SHA、manifest内file SHA、scientific contract、
     terminal gate / summaryを照合する。
  2. exp416 well audit / candidate identityとexp226 identity / foldだけを読み、
     outcome前に6診断、outer-4-fold ECDF、2 score、2中央値、4 cellsを作る。
  3. `regime_feature_freeze`、`regime_assignment`、`row_scope_freeze`のschema /
     logical SHAを保存してledgerをfreezeする。
  4. freeze後だけexp226 truth、exp072 control、exp416 by-well / persistent episode
     outcomeを結合し、pooled / by-well parityを確認する。
  5. pooled / fold / regime / fixed position / individual diagnostic / episode表、
     fold別SHA256 seedによる4096回within-fold置換、technical/scientific AND gateを作る。
- `tests/test_exp422_roughening_x10_failure_regime_attribution_readout.py`に9 testsを追加した。
- exp416 compactとの比較:
  - exp416: 12章 / 2,352行
  - exp422: 11章 / 2,376行
  - exp422はPF生成章を持たない代わりにsource contract、target-free freeze、
    association / permutation、attribution gateをNotebook上へ展開しており、
    同一exp helper importだけの薄い構成ではない。
- `Path(__file__)` / `__file__`はcompact sourceに残していない。
- compact実装のscientific contract SHAは
  `20d2644085334ed0028ff8ca0caa38d6379073980f3547c6d05b1f7eee410426`。
- 正規Notebook採用、Kaggle package / push / audit run、inference、submissionは
  行っていない。

## 実行量

- saved-output readout contracts: 1
- new prediction rows: 0
- scientific PF variants / PF well-runs / control reruns: `0 / 0 / 0`
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`
- reporting folds: 5

## 2026-07-28 CPU audit実行承認

- ユーザーの`実行してください`を、正規train Notebook採用、Kaggle package、
  canonical kernel push、CPU audit run、完了監視、結果記録の明示承認として記録した。
- inference / submissionは承認範囲に含めず、引き続き禁止する。
- canonical kernel:
  `kentookumura/exp422-rough-x10-regime-attribution-train`
- title: `exp422 rough x10 regime attribution train`
- 実行量を再確認した。
  - saved-output readout: 1
  - scientific variants / new prediction rows: `0 / 0`
  - PF well-runs / parent control reruns: `0 / 0`
  - LightGBM configs / trained folds / boosters: `0 / 0 / 0`
  - HMM / Beam / GPU: `0 / 0 / 0`
  - reporting folds: 5
- 実行承認反映直後、旧contract testが`run is unapproved`を期待して1件FAILした。
  科学式や実行量の不一致ではなく承認状態fixtureの陳腐化であり、現在の承認済み状態と
  承認フラグ欠落時のfail-closedを両方確認するtestへ更新した。
- 正規train Notebookをcompact self-contained sourceから採用した。
- canonical train packageを作成した。configの`kaggle_package_created`とstageを
  更新したため、push前に同じcanonical id / titleでpackageを再生成してbootstrapを
  最新configへ同期する。
- canonical packageのmetadata / bootstrapを確認した。
  - id / title:
    `kentookumura/exp422-rough-x10-regime-attribution-train` /
    `exp422 rough x10 regime attribution train`
  - private / CPU / internet-off / run-on-push: `true / true / true / true`
  - kernel sources: exp416 merge、exp072 control、exp226 reportingの固定3件
  - embedded config: `package_ready_pending_push`、audit承認true、GPU false
- Kaggle private CPU kernel version 1をpushし、実行を開始した。
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp422-rough-x10-regime-attribution-train`
- push後の`kaggle kernels pull -m`で同じcanonical idを確認した。
  - id_no: `128921651`
  - version: 1
  - private / GPU / internet: `true / false / false`
  - kernel source 3件とcompetition sourceを確認した。

## 2026-07-28 version 1 technical ERROR

- version 1は約`108.5 sec`でERRORになった。
- failure:
  `ValueError: exp416 prediction logical content SHA mismatch`
- 原因:
  - exp416の`freeze_prediction_frame`がlogical SHAへ使った列は
    `id / well_id / row_idx / likpf_roughening_x10_mean`の4列。
  - exp422 version 1は安全診断列を含む8列全体で、その4列SHAと比較していた。
  - 入力raw/decompressed SHA、source manifest、scientific contract、regime式、
    outcome、PF/model実行量の問題ではない。
- 修正:
  - `PARENT_LOGICAL_COLUMNS`を親と同じ4列へ固定した。
  - 8列全体は親schema SHA、およびexp422のfeature / assignment / row-scope
    logical SHAで引き続き監査する。
  - 専用testを追加し、親logical列契約を固定した。
- scientific config、2 score、cell、gate、入力version、実行量は変更していない。
- version 1でPF / model / booster / new predictionは各0。
- 修正後はJupytext test、`py_compile`、Ruff、専用`10 tests`、strict
  experiment validationをPASSした。
- canonical package内の`PARENT_LOGICAL_COLUMNS`、embedded config version 2、
  CPU / internet-off / run-on-push、固定3 kernel sourcesを確認した。
- 同じcanonical kernelへversion 2をpushし、実行を開始した。

## 2026-07-28 version 2 COMPLETE

- canonical private CPU kernel version 2は`COMPLETE`。
  - id_no: `128921651`
  - runtime: `362.8773446083069 sec`
  - peak RSS: `3.298248291015625 GiB`
  - generated at: `2026-07-28T11:13:45.776106+00:00`
- 3,783,989 rows / 773 wells、reporting folds 0--4を完走した。
- execution countは事前契約と一致した。
  - saved-output readout: 1
  - new prediction / PF / control rerun / LightGBM / booster / HMM / Beam / GPU: 0
- technical gateはPASSした。
  - 全入力SHA、source kernel version / id_no、identity、row / well / fold数を確認。
  - candidate / control pooled RMSE差は各0、by-well最大差は
    `7.105427357601002e-15 ft`以下。
  - truth / control / by-well / episode outcomeを開く前にregimeをfreezeし、
    outcome rowsはすべて0だった。
- feature freeze:
  - feature logical SHA:
    `0fed1d9ed954f6e585f6b8bcdd60c966bc58c67efd6660f7adadb5bbd97dccf4`
  - assignment logical SHA:
    `e459e3a438511710e48c8646ea1f283f54b1310729cd5e99aee9addd8e7b2fb2`
  - row-scope logical SHA:
    `242562b0d7a05fef042db29ad41b736991548825aebe26b773e759cc29fd194c`
  - target cell: 242 wells
- scientific gateはFAILした。
  - recovery pressure: pooled rho `-0.1666976968`、期待したpositive folds
    `0/5`、one-sided permutation p `1.0`。
  - damage exposure: pooled rho `-0.0414847532`、negative folds `4/5`、
    one-sided permutation p `0.1113009519`。
  - target-cell row RMSE gain `-1.8524495837 ft`、改善`1/5` folds。
  - target-minus-rest equal-well gain `-0.5189655682 ft`、
    improved-well fraction `0.3140495868`。
  - target cellのpersistent-offsetは4 wells / 4 episodes / 14,827 rowsで、
    SSE reductionは`45.8019674812%`だったが、全positive episode reductionの
    shareは`39.4006166985%`で事前閾値50%に未達。
- decision:
  `no_reproducible_target_free_regime_close_attribution_branch`
- exp416の`roughening_x10_rejected_close_without_rescue`は変更しない。
- score / transform / threshold / cell / roughening parameterのsame-OOF救済、
  adaptive policy、inference、submissionは行わない。
- artifact manifest SHA:
  `c2fe9339994e8785bf33dc0585f985d5a819e1ff6bf653262bad46e108c04f16`
- Kaggleログと`kernels files`でgate / summary / manifestを含む出力を確認できたため、
  output archive全体はダウンロードしていない。
- 完了後の`kaggle kernels pull -m`でもcanonical id / id_no、private、CPU、
  internet-off、固定3 kernel sources、competition sourceを再確認した。

## コマンドログ

実行済み:

```bash
make new-steering EXP=exp422_roughening_x10_failure_regime_attribution_readout
make new-exp EXP=exp422_roughening_x10_failure_regime_attribution_readout SOURCE=templates/experiment
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp422_roughening_x10_failure_regime_attribution_readout/exp422_roughening_x10_failure_regime_attribution_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp422_roughening_x10_failure_regime_attribution_readout/exp422_roughening_x10_failure_regime_attribution_readout_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/exp422_roughening_x10_failure_regime_attribution_readout/exp422_roughening_x10_failure_regime_attribution_readout_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp422_roughening_x10_failure_regime_attribution_readout/exp422_roughening_x10_failure_regime_attribution_readout_compact_selfcontained_train.py tests/test_exp422_roughening_x10_failure_regime_attribution_readout.py --select F821
.venv/bin/pytest -q tests/test_exp422_roughening_x10_failure_regime_attribution_readout.py
make validate-exp EXP=exp422_roughening_x10_failure_regime_attribution_readout
make prepare-kaggle-notebooks EXP=exp422_roughening_x10_failure_regime_attribution_readout EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp422-rough-x10-regime-attribution-train --title 'exp422 rough x10 regime attribution train' --run-on-push --strict"
kaggle kernels push -p experiments/exp422_roughening_x10_failure_regime_attribution_readout/kaggle/train
kaggle kernels status kentookumura/exp422-rough-x10-regime-attribution-train
kaggle kernels logs kentookumura/exp422-rough-x10-regime-attribution-train
kaggle kernels files kentookumura/exp422-rough-x10-regime-attribution-train
kaggle kernels pull kentookumura/exp422-rough-x10-regime-attribution-train -p /tmp/exp422-kaggle-pull-v2-final -m
```

結果:

- Jupytext test: PASS
- `py_compile`: PASS
- Ruff F821: PASS
- dedicated tests: `10 passed`
- strict experiment validation: PASS
- Kaggle version 2: COMPLETE
- technical / scientific gate: PASS / FAIL

## 再現性メモ

- seed policy: permutationだけfold別stable SHA256 seed
- stochastic components: fixed 4096 within-fold permutationsだけ
- CPU/GPU runtime: CPU `362.877 sec` / peak RSS `3.298 GiB`、GPU 0
- source kernel: exp416 merge version 2 / id_no 128912230
- input SHA: exp416 manifest、exp072 decompressed、exp226 decompressedを固定
- feature SHA: outcome attachment前のfeature / assignment / row scope logical SHAを記録
- prediction / model / submission SHA: 新規生成なし
- deterministic anchor: 推論・提出anchorとは扱わない

## 次のアクション

1. exp422内の救済を行わず、attribution branchをterminal closeする。
2. exp416のterminal FAILを維持する。
3. inference / submissionは行わない。
