# SESSION_NOTES

## 2026-08-02 設計確定

- ユーザー依頼により、区間selectorの新枠組みとしてrank学習を設計した。
- 実験名を`exp504_h512_regret_weighted_block_rank_selector`、routeを`ensemble`とした。
- exp293 fixed12 bank、H512 non-overlap blocks、exp264 corrected 88列を固定入力にした。
- RankNet型pairwise logistic、regret weight、Borda、固定anchor guardを1構成に限定した。
- 将来実行量を1 variant / 1 config / 5 outer folds / 5 CPU modelsとした。
- control再学習、candidate再生成、PF/HMM/Beam、GPUは0とした。
- 成功条件はpooled gain 0.05 ft、4/5 folds、固定scope、by-well p95/worstの全AND。
- FAIL時にhorizon/loss/weight/thresholdをsame-OOF救済しない方針を固定した。
- steering、backlog、experiment scaffoldだけを作成した。科学的実装、test、Kaggle package、
  train/inference/submissionは未着手・未承認。

## 固定証拠

- candidate bank content SHA:
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
- H512 block assignment decompressed SHA:
  `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`
- truth content SHA:
  `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`
- exp264 corrected 88-feature schema SHA:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`

## 次アクション

正規Notebook採用、Kaggle CPU package作成、push/runは別途承認後に行う。

## 2026-08-02 実装

- ユーザーの「exp504を実装してください」を凍結済み科学契約の実装承認として受けた。
- 既存の正規train/inference placeholderは上書きしていない。
- 別名Jupytext percent source
  `exp504_h512_regret_weighted_block_rank_selector_compact_selfcontained_train.py`と候補`.ipynb`を作成。
- self-contained sourceに次を実装した。
  - exp263 manifest/catalog/partition SHAを確認し、6 primitiveと固定formulaからfixed12を再構成。
  - exp293 block assignment raw/decompressed SHA、3,783,989 rows、773 wells、7,787 H512 blocksを確認。
  - corrected exp264 schema file/logical SHA、88列、candidate順、ordinal ID禁止を確認。
  - raw horizontal allowlist `MD/X/Y/Z/GR`とtypewell summaryから88列をtarget-freeに再生成。
  - `ctx__` 22列をcandidate間同値guard後にshared集約、残り66列をcandidate-specific集約。
  - 固定9統計によりcandidate block 594列、shared block 198列、block context 6列を生成。
  - pair表現`[left-right, abs(left-right), mean, shared, block]`を1,986列に固定。
  - outer-train block MSE、tie除外、`row_count * log1p(abs(MSE差))`、fold mean 1正規化、
    ordered pair両方向とhalf weightを実装。
  - 1 config / 5 outer folds / 5 CPU `LGBMClassifier`、800 trees固定、early stoppingなし。
  - 両方向確率反対称化、Borda、tie時anchor-first、strict `p>0.5` anchor guardを実装。
  - outer-valid prediction freeze後のtruth readout、fold/scope/by-well/rank/choice/switch gateを実装。
  - block/pair/model/prediction/feature/OOF生成物SHAとfeature importance mean/plotを実装。
- 非promotion readoutの曖昧さだけを固定した。
  - NDCG@1はcandidate MSE query-rankのlinear relevance `12-rank`を使う。
  - promotion gateにはNDCGを使わない。
- 将来実行量は`1 variant × 1 config × 5 folds = 5 CPU boosters`。
- control再学習0、candidate再生成0、PF/HMM/Beam/GPU/inference/submission各0。

### 親compactとのNotebook比較

- 親exp293 compact train: 1,963 lines / 8章。
- exp504 compact train候補: 2,352 lines / 9章。
- exp504は親のruntime/SHA、fixed12、block、truth-late、metrics/生成物の役割を維持し、
  exp264 feature再生成、H512集約、pair table、rank model、Bordaを独立章として追加した。
- 同一exp helper import、`Path(__file__)`、薄い`main()`だけの構成はない。

### 実行した検証

- `.venv/bin/python -m py_compile ...compact_selfcontained_train.py settings.py`: PASS。
- `.venv/bin/ruff check ... --select F821,F401,E501`: PASS。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...`: PASS。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`: PASS。
- `.venv/bin/pytest -q .../test_exp504_contract.py`: `9 passed`。
- `make validate-exp EXP=exp504_h512_regret_weighted_block_rank_selector`: strict PASS。
- 保存済み入力preflight: block `3,783,989 / 773 / 7,787`、file/decompressed SHA PASS、
  row schema `88`、candidate/shared/pair幅`594 / 198 / 1,986`を確認。
- `task validate-exp`は環境に`task`コマンドがなく実行不能だったため、規定のMakefile同等
  コマンドを使用した。科学実行失敗ではない。

### 実装時点の未実行

- ローカルnotebook/full OOFは実行していない。Kaggle Notebookを正とする。
- この実装時点では、正規Notebook採用、Kaggle package、push/run、model/OOF生成、
  inference、submissionは未承認だった。

## 2026-08-02 Kaggle train実行承認

- ユーザーの「実行してください」を、凍結済みexp504 trainの正規Notebook採用、Kaggle CPU
  package作成、push/run、完了監視、train-side OOF記録の承認として受けた。
- 実行量を再確認した。
  - scientific variant: 1
  - LightGBM config: 1
  - outer folds: 5
  - 合計: 5 CPU models / boosters
  - 親/control再学習、candidate再生成、PF/HMM/Beam、GPU: 0
- inference、submission、downstream昇格は今回の承認に含めない。
- Kaggle credential checkerでOAuth credentialとKaggle CLI用legacy credentialを確認した。
- compact self-contained train候補を正規train notebookへ採用する。正規inference notebookは
  placeholderのまま維持する。
- canonical kernelを
  `kentookumura/exp504-h512-regret-weighted-block-rank-selector-train`、titleを
  `exp504 h512 regret weighted block rank selector train`としてCPU/private/internet off/
  run-on-push packageを作成した。
- packageにはcompetition sourceと固定4 kernel sourcesを付与し、bootstrap manifest内の
  `config.yaml`、`settings.py`、`project.yml`を確認した。
- full-name由来の初回slug
  `exp504-h512-regret-weighted-block-rank-selector-train`は53文字で、Kaggle
  `SaveKernel 400 Bad Request`となり実行開始前に拒否された。直前のpullも403でkernel作成を
  確認できなかった。固定4 input kernelは個別metadata pullがすべて成功した。
- repo内で反復確認済みのKaggle 50文字上限に合わせ、科学契約と実験番号を変えず、
  canonical id/titleを44文字の
  `kentookumura/exp504-h512-regret-block-rank-selector-train` /
  `exp504 h512 regret block rank selector train`へ同時に短縮する。
- 2026-08-02 14:40:34 UTC、短縮canonical packageをpushし、Kaggle private CPU version 1を
  開始した。metadata pull成功、`id_no=129488458`、GPU/internet false、固定4 kernel sourcesと
  competition sourceの反映を確認した。

## 2026-08-02 Kaggle train version 1完了

- canonical kernel: `kentookumura/exp504-h512-regret-block-rank-selector-train`
- version / id_no: `1 / 129488458`
- Kaggle worker: `COMPLETE`
- runtime / peak RSS: `5,422.757684 sec / 12.883976 GiB`
- actual execution: `1 variant / 1 config / outer 5 / 5 CPU models / 5 boosters`
- 親control再学習、candidate再生成、PF/HMM/Beam、GPU、inference、submission: 全0
- technical gate: 全PASS。3,783,989 rows、773 wells、7,787 blocks、入力/schema SHA、
  truth-late ledger、prediction freeze、5 model SHA、OOF SHAを確認した。

### 科学結果

- pooled selector / anchor RMSE: `8.114276980 / 8.238331546 ft`
- pooled delta: `-0.124054566 ft`、pooled gain gateはPASS。
- fold delta: `[-0.352783, -0.275423, -0.235025, +0.123583, +0.100364] ft`。
  nonworseは`3/5`でFAIL。
- hidden-like spatial / typewell-purged delta:
  `+0.285759 / +0.269833 ft`で固定`+0.02 ft`上限をFAIL。
- by-well改善 / 悪化: `446 / 319`、p95 `+2.963656 ft`、worst
  `81bf5923 +16.799044 ft`で固定`+0.25 ft`上限をFAIL。
- H512 top-1 / weighted pair accuracy / NDCG@1 / top3 coverage:
  `0.112624 / 0.741908 / 0.682537 / 0.286503`。
- anchor選択`2,942 / 7,787 = 0.377809`、guard fallback`747 = 0.095929`、
  inter-block switches `4,427`。

### 再現性と取得物

- executed config file SHA:
  `26506f78320117d2c628b0fd5a840e6021a47545614cd126e2d9975b3ff108ba`
- pushed notebook / kernel metadata SHA:
  `d2c6ecc6a7842a23fee0886b58e6a87497594b7c7f787e57f1a46883f9390c39` /
  `484f94975126655ea6cb589f38f2ede2200e7a401ba7bf5fed55de5c44f1e4e1`
- block feature content SHA:
  `f333d097d8bdd369b2b6786328dee050d6bb5ba4114d810e26e60be976fd56c8`
- model manifest logical SHA:
  `60696a0574de0c62f8c413c2344a664f40a40634f202ec3a02a754bd2ef3de25`
- OOF prediction content SHA:
  `1dd09844b70536ec7eae26d6656efb70a00bdc3488a57aca188fb6dfc3b2504f`
- CV判定はKaggle logsを根拠にした。再現性SHA、model manifest、by-well/scope/rank表の
  実ファイル確認が必要だったため、output archive全体ではなく`--file-pattern`で必要な
  小型生成物だけを`kaggle/output/train_v1/artifacts/`へ取得した。
- sklearnのfeature-name warningはNumPy予測入力に対する警告で、全5 foldsの予測、SHA、
  readout、technical gateは完了しているため科学実行失敗ではない。

### 判断

promotionは`FAIL_TERMINAL_CLOSE_WITHOUT_HORIZON_LOSS_WEIGHT_OR_THRESHOLD_RESCUE`。
pooled平均改善だけでfold / hidden-like / well-tail FAILを上書きしない。凍結契約どおり、
exp504内のhorizon/loss/weight/model/threshold/guard/smooth/blend/gate救済、再実行、inference、
submissionは行わず終端閉鎖する。

原因確認が将来必要な場合だけ、保存済みOOF/block/rank生成物を使う0-model・0-predictionの
block-rank tail attributionを別steering・別承認の低優先P4として検討する。
