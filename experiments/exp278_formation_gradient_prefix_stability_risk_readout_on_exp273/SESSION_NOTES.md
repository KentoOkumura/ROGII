# exp278 formation gradient prefix stability risk readout on exp273 セッションノート

## 目的

backlog `formation_gradient_prefix_stability_risk_readout_on_exp273`を、exp273の保存済みcandidate
outcomeに対する0-booster target-free prefix stability readoutとして実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 1実行承認済み、canonical push準備中
- CV / LB: diagnostic-only / 対象外
- inference / submission: disabled

## 実行コスト契約

- active variant / model config / trained fold / booster: `0 / 0 / 0 / 0`
- parent/control再学習: 0
- HMM path生成: 0
- fixed candidate: exp273 gradient 5本、3,783,989 rows / 773 wells
- plane再計算: full / last-512 / last-256、773 wells、CPU逐次
- GPU / internet / inference / submission: off / off / disabled / disabled

## 固定入力

- exp273 plane diagnostics SHA:
  `484ced57a3c0a3e19b4e1002747c326f576d98a439a9a6418bf793bac9c1a665`
- exp273 by-well metrics SHA:
  `f54d2b6dc321a5c9add65accc9e2bab440c3429c0d80354d85246a6375c7b52d`
- shard 0 raw / decompressed SHA:
  `acb943b7...3eb2c1` / `347b8755...eac97`
- shard 1 raw / decompressed SHA:
  `ae78cfc1...e7d9e48` / `98939d08...7d407`
- raw horizontalは`MD/X/Y/Z/TVT_input`だけを読み、file SHA manifestをKaggle実行時に保存する。

## 実装

- `.steering/20260718-exp278-formation-gradient-prefix-stability-risk-readout-on-exp273/`へ
  要件、設計、tasklistを作成した。
- self-contained Jupytext train sourceを9章で実装した。同じexp directoryのhelper importはない。
- exp273のSVD geometryとHuber IRLSを再実装し、full-prefixのvalidity/fallback/geometry/
  generation gradient/plane RMSEを保存済みdiagnosticsへparityする。
- 実データ3 wellsの限定関数確認で、full-valid wellでもtail windowはgeometry guard不通過になりやすく、
  exp273 fallbackをそのまま使うと角度・大きさ・RMSEがゼロ/欠損になることを確認した。
  generation guardは変更せず、min-points/rank-2を満たすwindowの同一Huber diagnostic fitを別列にした。
- angle、magnitude log-ratio、plane RMSE log-ratio、rank absolute gap、condition log-ratio、
  validity flipの6成分をpair最大・等重み平均へ固定した。
- well foldは`sha256("exp278::outer_fold::<well>") % 5`。exp273 full-valid分布は
  `19 / 20 / 28 / 24 / 20 wells`で、事前min 15を満たす。
- shard gzipを250,000 rowsずつ読み、candidate RMSEをwell集約してaggregate by-well metricsへparityする。
- feature frameはoutcome-like列を拒否しlogical SHAを固定した後だけ、bank-mean / bank-max /
  candidate-specific delta RMSEへjoinする。
- inference notebookはdiagnostic-only disabled contractで、submissionを生成しない。

## Notebook構造比較

- 親exp273 canonical train source: 2,037行 / 10章。
- exp278 train source: 約1,250行 / 9章。
- exp278はruntime/contract、input SHA、shard outcome parity、Huber plane、stability risk、
  5-fold readout、guard、metrics/SHA保存をセル上へ展開した。
- HMM kernel/shard generation章は本readoutで再実行しないため削除した。薄い`main()`呼び出しではない。
- train/inference sourceの`__file__`依存: 0。

## コマンドログ

```bash
make new-steering EXP=exp278_formation_gradient_prefix_stability_risk_readout_on_exp273
make new-exp EXP=exp278_formation_gradient_prefix_stability_risk_readout_on_exp273
.venv/bin/pytest -q experiments/exp278_formation_gradient_prefix_stability_risk_readout_on_exp273/tests/test_exp278_formation_gradient_prefix_stability_risk_readout.py
# 7 passed
.venv/bin/python -m py_compile <exp278 train.py> <exp278 inference.py> <exp278 test.py>
.venv/bin/ruff check <exp278 sources and test>
# All checks passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <exp278 train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <exp278 train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <exp278 inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <exp278 inference.py>
make validate-exp EXP=exp278_formation_gradient_prefix_stability_risk_readout_on_exp273
# strict validation passed
make validate-template
make test
# repository 154 tests passed
make prepare-kaggle-notebooks EXP=exp278_formation_gradient_prefix_stability_risk_readout_on_exp273 \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp278-gradient-prefix-stability-readout-train \
  --title 'exp278 gradient prefix stability readout train' --strict"
```

ローカルnotebook/full readoutは実行していない。実装確認として合成dataのunit testsと、raw train 3 wellsの
plane parity関数だけを限定確認した。

## Kaggle package

- kernel: `kentookumura/exp278-gradient-prefix-stability-readout-train`
- metadata: private、CPU、GPU/TPU off、internet off、run-on-push false
- competition source: ROGII competition
- kernel sources: exp273 aggregate / shard0 / shard1 の3件
- config source/package/bootstrap SHA:
  `d3c8c9cb5526a54eae59a28d55fbd9ba3a739551085f429056a0b9faf735c525`
- train source/package/bootstrap SHA:
  `6dc1e72ea1e22eafa02ca693018783201ec89d118b6846b73fbe0f464d2ad0de`
- prepared notebook SHA:
  `e099d9f201e9fd76cb9b7b808ad47ac032cde4eea830a55b0f04a8650bed84b4`
- configは`run_approved=false`、0 booster、HMM生成0。Kaggle pushは行っていない。

`scripts/review_exp_docs.py`は現在のrepositoryに存在しなかったため実行できなかった。代わりに
`make validate-exp`のstrict docs validation、README/result/metrics/steeringの目視確認を完了した。

## 再現性メモ

- RNG: なし。bootstrapなし。
- fold: stable SHA256 well hash。
- compute: Kaggle CPU single process、well文字列順。
- gzip: raw/decompressed SHAを分離してhard guardする。
- output: input manifest、plane/stability logical SHA、全CSV/plot file SHAを保存する。
- model / prediction / submission SHA: 新規生成なしのため対象外。
- deterministic anchor: route/CV/submission anchorではない。rerun前はdiagnostic anchorとも呼ばない。

## 次のアクション

1. 0 variant / 0 config / 0 trained fold / 0 booster / HMM path 0を提示し、Kaggle CPU push承認を得る。
2. PASSでも別gate設計までinferenceしない。FAILならsingle component/window/clip/gridで救済せずbranchを閉じる。

## 2026-07-18 Kaggle CPU実行承認

- 承認時刻: `2026-07-18 13:56:37 JST`
- 承認scope: canonical private train kernelを1回pushし、CPU readoutを完了まで監視する。
- 実行量: 0 variant / 0 model config / 0 trained fold / 0 booster / HMM path生成0。
- parent/control再学習: 0。exp273 aggregate/shardの保存済み生成物だけをoutcomeに使う。
- runtime: Kaggle CPU、GPU/TPU off、internet off、single process。
- inference / submission: disabled。本承認にはraw-test portと提出を含めない。
- credential: OAuth CLIとlegacy CLI credentialを確認。API tokenは未設定だがKaggle CLI操作には影響しない。
- remote preflight: canonical kernel pullは403、`kernels list --search exp278 --mine`は`Not found`。
  既存kernelは確認されず、別slugを増やさず初回canonical pushへ進む。
- approved package config SHA: `55d6f2338f78340b39611a535ed22a376ea6b9d33cde312318bd0f372cdf8255`。
  source/package/bootstrap manifestで一致した。
- approved train source SHA: `6dc1e72ea1e22eafa02ca693018783201ec89d118b6846b73fbe0f464d2ad0de`。
  source/package/bootstrap manifestで一致した。
- approved prepared notebook SHA: `97018462dd12e467fc36a088d13c045284a46aebe02456a028ef96bd52de1f03`。
- metadata: canonical id/title slug一致、private、CPU、GPU/TPU/internet off、run-on-push true、
  competition source 1、exp273 kernel source 3。

## 2026-07-18 Kaggle CPU version 1 push

- push時刻: `2026-07-18 13:58:10 JST`
- kernel: `kentookumura/exp278-gradient-prefix-stability-readout-train`
- version: 1
- Kaggle `id_no`: `127738648`
- push成功後の`kernels pull -m`でcanonical id/title、private、CPU、GPU/TPU/internet off、
  competition source 1、exp273 kernel sources 3を確認した。
- Kaggle正規化後notebook SHA: `1eff96f6f814b28c7d4c9a2599b6b592f520601867609a8aa006fc0f7203a93c`。
- push成功直後にlocal `execution.run_approved=false`へ戻し、version 2の誤pushを防止した。

## 2026-07-18 Kaggle version 1 technical no-op

- Kaggle statusは`COMPLETE`、logs終端は約24.6秒だったが、bootstrap 15 files以外のstdoutがなく、
  outputにはbootstrap support filesだけで`artifacts/`生成物が0件だった。
- 原因: notebook runtime判定の`"__IPYTHON__" in globals()`がKaggle cellでfalseとなり、
  `if EXECUTE_NOTEBOOK:`配下の全readout cellをskipした。
- 影響: raw train、exp273 shard/outcome、Huber plane、risk、guardは未実行。学習/HMM/boosterも0。
  version 1は科学結果ではなくtechnical no-opとして扱う。
- 修正: train/inference sourceの判定だけを`get_ipython() is not None`へ変更し、通常Python importでは
  false、Jupyter/Kaggle kernelではtrueとなるcontract testを追加した。
- feature、window、risk、fold、outcome、guard、入力SHA、実行量は変更しない。
- ユーザーの実行依頼を完遂するため、同じcanonical kernelのversion 2 technical retryに限って
  `execution.run_approved=true`へ再設定した。

## 2026-07-18 Kaggle CPU version 2 retry package

- package確定時刻: `2026-07-18 14:03:12 JST`
- retry scope: version 1のruntime判定だけを修正した同一canonical kernelのtechnical retry。
- 実行量: 0 variant / 0 model config / 0 trained fold / 0 booster / HMM path生成0。
- config source/package/bootstrap SHA:
  `ad1454e5ee6f1bd6e7ee7e503d997c2d70854ec2361f9743c27f584cfb1bef96`。
- train source/package/bootstrap SHA:
  `4603197a4a2030f39cd70c694289a33f218023d6cfdde0ae6e132ba94c195f10`。
- inference source/package/bootstrap SHA:
  `9bedb2e1753998c835f02bc205773bfde163751d38b7d5c0c424d46903252665`。
- prepared notebook SHA:
  `edbbafbfab33a81fb2d9031b5306db8fbee9fb59afac8db42553c07931b2cc46`。
- package notebook内にも`get_ipython() is not None`を確認した。
- metadata: canonical id/title、private、CPU、GPU/TPU/internet off、run-on-push true、
  competition source 1、exp273 kernel sources 3。
- `make validate-exp`、Jupytext round-trip、Ruff、py_compile、targeted tests 7件を再通過した。

## 2026-07-18 Kaggle CPU version 2 push

- push時刻: `2026-07-18 14:03:45 JST`
- kernel: `kentookumura/exp278-gradient-prefix-stability-readout-train`
- version: 2
- Kaggle `id_no`: `127738648`（version 1と同じcanonical kernel）。
- push成功直後にlocal `execution.run_approved=false`へ戻し、version 3の誤pushを防止した。

## 2026-07-18 Kaggle CPU version 2 result

- status: `COMPLETE`。readout summary出力は約49.6秒、notebook/HTML変換を含むlogs終端は約56.7秒。
- Kaggle正規化後notebook SHA:
  `41f1466ddf64e91d9072b2e47388907aac6ed108dfd6b782093e31cd640e2e6b`。
- runtime contract: 0 variant / 0 LightGBM config / 0 trained fold / 0 booster / parent control再学習0 /
  HMM path生成0 / GPU off / internet off / inference off / submission off。
- coverage: 3,783,989 rows / 773 wells / exp273 full-valid 111 wells。
- full-valid fold wells: `19 / 20 / 28 / 24 / 20`。
- full-plane parity: gradient/geometry/RMSEの6項目すべてPASS。
- candidate RMSE parity: 5 candidates x 773 wells = 3,865件すべてPASS、最大絶対差
  `8.277822871605167e-13 ft`。
- primary fold Spearman: `0.059649 / 0.177444 / 0.125889 / -0.123478 / -0.061654`。
- primary pooled Spearman: `0.07424534924534924`（positive guard PASS）。
- positive folds: `3/5`、required `5/5`（guard FAIL）。
- q0 mean bank delta RMSE: `-2.195777569511029 ft`。
- q4 mean bank delta RMSE: `+2.1576941433809207 ft`（q4 > q0 guard PASS）。
- candidate別pooled Spearmanは`0.073438`〜`0.075834`、bank-maxは`0.077843`。
  全てfold 3/4が負方向で、report-only結果はprimary guardを救済しない。
- frozen stability feature logical SHA:
  `4d03bf82a5f5b8775661deaa7d544c97ff11dfda6ab422c68fa10efb0ba47f08`。
- readout summary file SHA:
  `7060c20e7de32e3ca5db6f2b0bd673838708f0c9f635070cd154a6f0ef352b74`。
- reproducibility manifest file SHA:
  `c99312a80f4f5c7c8f8d11451dc5eb38219a198b35b66b3ff4bcc864765d3db1`。
- summary記載の8 artifact SHAは取得ファイルと全件一致し、summaryとmanifestのartifact記録も一致した。
- decision: `readout_guard_failed_branch_close`。component/window/clip/weight/threshold grid、別gate、
  HMM再実行、raw-test inference、submissionを行わず、exp273 formation-gradient branchを閉じる。
- 成功runはversion 2の1回だけなのでdeterministic diagnostic anchorとは呼ばない。CV/LB anchor更新なし。

## 最終検証

- exp278 targeted tests: 7 passed。
- exp278 strict validation、Ruff、py_compile、train/inference Jupytext round-trip: PASS。
- repository全体rerun: 154件中153 passed。失敗1件は別実験exp277のtestが
  `execution.run_approved=true`を要求する一方、現在のexp277 configがfalseである既存不整合。
  exp278とは無関係なためexp277は変更していない。
