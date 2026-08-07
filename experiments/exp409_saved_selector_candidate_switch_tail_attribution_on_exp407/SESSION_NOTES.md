# exp409_saved_selector_candidate_switch_tail_attribution_on_exp407 セッションノート

## 目的

exp407のtail悪化を、corrected exp264からexp407へのhard-selected candidate遷移へ
保存済みcandidate-score OOFだけで帰属する。改善案の探索やexp407救済は行わない。

## 現在の状態

- Route: `ml_model`
- 状態: private CPU v1完了・technical PASS・tail consistency gate FAILで閉鎖
- model / booster / prediction: 0 / 0 / 0
- CV / LB: 対象外 / 未実施
- Kaggle package / run: canonical version 1 COMPLETE
- inference / submission: 禁止

## 変更点

モデルや予測surfaceは変更しない。保存済み親/exp407 OOFのhard selection差だけを
truth-freeにfreezeし、freeze後にSSE差を診断集計する新しいreadoutを追加した。

## 2026-07-26 設計確定

- 親はexp407、比較対象はcorrected exp264 Stage B v5。
- 両candidate-score OOFのSHA、12候補順、11候補hard domain、foldを固定した。
- Phase 1は`pred_abs_error`とtarget-free列だけを読み、両hard selection、
  transition、distance、hidden-like roleをfreezeする。
- freeze fileとmanifest SHAを作るまで`actual_abs_error`を読まない。
- Phase 2だけで`actual_abs_error`を読み、candidate value/error parityを照合して
  additiveな`exp407 squared error - parent squared error`を計算する。
- attributionはtransition、fold、distance、hidden-like、well別に保存する。
- concentration gateはmagnitude thresholdを置かない。同じpositive excess-SSE
  rank-1 transitionが1000+とhidden-like 2面で各4/5 folds、
  固定worst well `52f1e77a`でもrank-1の場合だけ原因支持とする。
- 結果にかかわらずexp407のscientific FAILは再分類しない。

## 入力確認

- exp407 v1 kernel
  `kentookumura/exp407-inverse-rmse-dual-selector-exp264-train`には
  `candidate_score_oof.parquet`が存在する。
- 親corrected exp264 Stage B v5 OOF
  `9a91b625...d48a`は
  `experiments/exp264_exp263_candidate_confidence_dual_selector/kaggle/output/stage_b_v5/artifacts/`
  にローカル保存済み。
- 現在のexp264 Kaggle kernel最新版はStage C出力であり、corrected Stage B v5の
  `candidate_score_oof.parquet`をkernel sourceとして直接提供していない。
- 親OOFをprivate Dataset
  `kentookumura/exp409-exp264-stage-b-v5-oof-input`へ作成した。
  アップロード前後のローカルSHAは
  `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`
  で一致している。

## 実行予算

| variant | model | fold fit | booster | PF/HMM/Beam | prediction | control再学習 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 再現性

- RNG: なし
- runtime: CPU single process
- parent OOF SHA:
  `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`
- exp407 OOF SHA:
  `d993b806d92c2462c1509f110669b272b27d48806c0280a2cf54e87c7f32f1e8`
- hidden-like assignment SHA:
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- Phase 1 freeze、truth ledger、Phase 2 row attribution、summary SHAを実行時に記録する。
- deterministic anchor: true。private Dataset / kernel source、kernel version 1、
  id_no、入力SHA、freeze / row attribution / summary SHAをpostrun記録した。

## 次

1. exp409を完了済みとしてstrategy backlogから削除する。
2. exp407を再開せず、same-OOF selector rescueを行わない。

## 2026-07-26 private CPU v1実行承認

- ユーザーの「実行してください」を、親OOF private Dataset作成、
  canonical Notebook採用、private CPU Kaggle v1 push/runの承認として記録した。
- local Notebook実行、inference、submissionは承認対象外であり、引き続き行わない。
- 実行数はvariant 0、LightGBM config 0、fold fit 0、booster 0。
- canonical kernel id:
  `kentookumura/exp409-selector-switch-tail-attribution-train`
- 入力はexp407 kernel sourceと、上記のexp264 private Datasetだけに固定する。

### pre-execution push rejection

- 初回packageの長いID
  `kentookumura/exp409-saved-selector-candidate-switch-tail-attribution-on-exp407-train`
  はKaggle `SaveKernel 400`で拒否された。Notebook実行は開始していない。
- 同じ長いIDへの`kaggle kernels pull -m`は403で、kernelが作成されていないことを
  確認した。
- 既知の長slug制約として、実験番号と診断の意味を保つ45文字の
  `exp409-selector-switch-tail-attribution-train`へid/titleを一致させて
  canonical名を確定した。科学contract、入力、runtime、実行数は変更しない。

### canonical v1 push

- `kentookumura/exp409-selector-switch-tail-attribution-train` version 1を
  private CPU / internet off / run-on-pushで開始した。
- Kaggle kernel id_no: `128678587`。
- push成功時の埋め込みconfig SHA:
  `79735dbc65f4d1ddb30188da2598cdc5768d318fdc531cdcd15d9e15a21dd1f1`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp409-selector-switch-tail-attribution-train`

## 2026-07-26 canonical v1完了

- status: `COMPLETE`。gate出力まで約`179.259 sec`。
- expected 3,783,989 rowsを処理し、switched rowsは1,289,588
  （34.0801%）、transition inventoryは121。
- 親OOF、exp407 OOF、hidden-like assignmentは実読込SHAが固定値と一致した。
- truth-free Phase 1の禁止truth readは0。freeze SHA確定後だけ
  `actual_abs_error`を読むledgerを確認した。
- overall RMSEはparent `8.587004`、exp407 `8.668141`、delta `+0.081137 ft`。
- scope deltaは1000+ `+0.091232`、hidden-like spatial `+0.103759`、
  typewell-purged `+0.079052 ft`。
- 固定worst well `52f1e77a`では
  `exp226_k16__selfgr_hmm_a070 -> likpf_mean__exact_hmm`がrank-1で、
  positive excess-SSE shareは約85.99%。
- ただし同遷移は1000+で1/5 folds、hidden-like 2面で0/5 foldsしかrank-1でない。
  全3 tail scope各4/5 foldsを満たす同一transitionは0件。
- gateは`passed=false`、
  decisionは`diffuse_or_nonreproducible_candidate_switch_cause`。
  exp407はscientific FAILのまま変更なし。
- small outputだけを取得し、manifest記載SHAと一致した。大きな
  `selection_freeze.parquet` / row attribution Parquetは取得していない。
- model / booster / prediction / inference / submissionは各0。

## 2026-07-26 implementation-only完了

- 1,329行・8章のcompact self-contained Jupytext train候補を作成した。
- 17セル（markdown 9、code 8）の別名Notebook候補へ変換した。
- 親exp407 compactは466行・8章。exp409は2つの約400--930 MB Parquetを
  二相stream処理し、freeze、truth parity、5系統の集計、gate、SHA manifestを
  self-containedに持つため1,329行となった。章の役割は欠落していない。
- 正規train Notebookはtemplate placeholderのまま上書きしていない。
- Phase 1ではtruth列をschema段階で拒否し、両surfaceのkey、12候補順、
  candidate value、11候補selection domain、schema/contract SHAを照合する。
- Phase 2はfreeze SHA確定後だけ`actual_abs_error`を読み、両surfaceの
  actual-error parityを確認してSSE差を集計する。
- transition overall / fold / distance / hidden-like / well、rank-1 4-of-5 gate、
  fixed worst-well照合、plot、truth ledger、reproducibility manifestを実装した。
- synthetic testはtruth早期read拒否、tie-break、key/value parity、
  physical freeze file、SSE加法性、4-of-5 gateを含む9件PASS。
- py_compile、Ruff F821/F811/F401/E501、Jupytext round-trip、
  strict `validate-exp`をPASSした。
- Notebookはローカル実行していない。Kaggle package/push/run、private dataset作成、
  inference、submissionも行っていない。

### 実行コマンド

```bash
make new-steering EXP=exp409_saved_selector_candidate_switch_tail_attribution_on_exp407
make new-exp EXP=exp409_saved_selector_candidate_switch_tail_attribution_on_exp407 \
  SOURCE=templates/experiment
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407_compact_selfcontained_train.py
.venv/bin/python -m py_compile \
  experiments/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407_compact_selfcontained_train.py
.venv/bin/ruff check \
  experiments/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407/exp409_saved_selector_candidate_switch_tail_attribution_on_exp407_compact_selfcontained_train.py \
  tests/test_exp409_saved_selector_candidate_switch_tail_attribution.py \
  --select F821,F811,F401,E501
.venv/bin/pytest -q \
  tests/test_exp409_saved_selector_candidate_switch_tail_attribution.py
make validate-exp \
  EXP=exp409_saved_selector_candidate_switch_tail_attribution_on_exp407
```
