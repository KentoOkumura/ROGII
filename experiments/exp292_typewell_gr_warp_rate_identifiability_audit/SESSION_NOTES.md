# exp292_typewell_gr_warp_rate_identifiability_audit セッションノート

## 目的

exp268の保存済み5本のinitial-rate candidateを、prefix-calibrated Type Well forward GRの
周波数・形状整合性からtarget-freeに順位付けできるかを監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 1完了・`FAIL_CLOSE_NO_RESCUE_GRID`
- CV / LB: 0-booster train-side audit完了 / 対象外
- inference / submission: disabled
- blocker: なし。事前登録した停止条件によりbranchを閉鎖済み

## 設計済み実行契約

- active audit variants: 1
- fixed candidates: 5 (`tail30 / w32 / w64 / w128 / w256`)
- score horizons: 3 (`H128 / H256 / H512`)、primary `H256`
- LightGBM config / trained fold / booster: 0 / 0 / 0
- evaluation folds: 5（metric安定性のreadoutのみ）
- HMM/PF well-runs: 0
- parent/control retraining or regeneration: なし
- GPU / raw-test inference / submission: なし / なし / なし

## 固定した設計

- known prefix末尾最大512行だけでrobust affineとresidual/derivative scaleをfitする。
- unknown suffixはType Well forward GRとhorizontal GRだけを使う。
- scoreはGaussian residual、NCC、chain-rule derivative residualの等重み1式。
- negative controlはstable SHA256 within-well circular-shuffle、no-GR controlは常時tail30 safe。
- primary H256 AUC lift `>=0.02`かつ4/5 folds正、top1 RMSE gain `>=0.10 ft`かつ4/5 folds改善、
  1000+ / hidden-like非悪化を全通過した場合だけPASSとする。
- FAIL時のrate/window/horizon/calibration/weight/threshold救済gridは禁止する。
- PASSしてもtop1 replacement、inference、submissionへ進まない。

## 再現性メモ

- `docs/06_reproducibility.md`を確認済み。
- real scoreはRNGなし。shuffleだけstable SHA256 per-well local RNGを使う。
- single process、well/candidate/horizon固定順。global RNGとPython `hash()`は禁止する。
- exp268/209 gzipはdecompressed content SHAを主証拠としてhard guardする。
- target-free score/selectionはschema/content SHAをtruth join前に固定する。
- model、selected row prediction、submissionは生成しない。
- deterministic submission anchorではなく、固定入力へのdeterministic diagnosticとして扱う。

## コマンドログ

### 2026-07-19 設計・scaffold作成

```bash
task new-steering EXP=exp292_typewell_gr_warp_rate_identifiability_audit
make new-steering EXP=exp292_typewell_gr_warp_rate_identifiability_audit
make new-exp EXP=exp292_typewell_gr_warp_rate_identifiability_audit
make validate-exp EXP=exp292_typewell_gr_warp_rate_identifiability_audit
```

- `task`は環境に存在せず起動前に失敗したため、同等の`make new-steering`へ切り替えた。
- steering: `docs/legacy/steering/20260719-exp292-typewell-gr-warp-rate-identifiability-audit/`
- experiment: `experiments/exp292_typewell_gr_warp_rate_identifiability_audit/`
- 親exp268、exp209 emission、exp288可視化、exp170/211/132 negative evidence、
  `docs/06_reproducibility.md`を確認した。
- notebookはtemplateのまま。実験ロジック、Jupytext source、test、packageは作成していない。
- YAML/JSON parseとstrict experiment validationはPASSした。これはscaffold/docs/configの検証であり、
  notebook実装や科学ロジックの検証ではない。
- local notebook/data実行、Kaggle prepare/push、output取得は行っていない。

## 次のアクション

exp268 aggregateが773 wells / 3,783,989 rows、5候補coverage/diversity、shard/aggregate SHAを
確定した後、canonical notebook採用とKaggle CPU package/pushの承認を得て1回だけ実行する。

### 2026-07-19 実装

- ユーザーの実装指示により、aggregate未完了でも実行前hard guardと合成テストを先行実装した。
- Kaggle read-only確認ではshard0/1はいずれも`COMPLETE`、aggregate kernelは未作成/未実行だった。
- shard0: 375 wells / 1,853,957 rows、decompressed SHA
  `a38ac16d12c9cd650170d16a9eb0b75159dd6e119443d33b7d7290a9e5347066`。
- shard1: 398 wells / 1,930,032 rows、decompressed SHA
  `30d6d7e930ffdec02f0da46108803c5640a03b26ea9b6cf8232ad7fdb06f0d36`。
- union期待値は773 wells / 3,783,989 rowsと一致する。ただしcandidate diversityとaggregate
  manifest/content SHAはaggregate実行まで未確定なので、train entrypointはfail-closedで停止する。
- compact train sourceはshardのtruth列を`usecols`で読まず、target-free score/selectionを保存・
  content/schema SHA固定してからraw horizontalの`TVT`だけを別loaderで読む。
- raw horizontal/typewell、aggregate/shard/control/hidden-like assignmentのinput SHAをmanifestへ記録する。
- compact inference sourceは常に`RuntimeError`で停止し、raw test、selected prediction、submissionを扱わない。
- 初回実装時は既存canonical `.ipynb`を上書きせずalternate compact候補として生成した。
- `experiments/exp292_typewell_gr_warp_rate_identifiability_audit/tests/test_exp292_typewell_gr_warp_rate_identifiability_audit.py`に11 contract testを追加した。
- Jupytext `--test`、py_compile、Ruff、strict experiment validation、template validationはPASSした。
- repository全体の`pytest -q`は276件PASSした。
- local実データ実行、Kaggle prepare/push、output取得、submissionは行っていない。

### 2026-07-19 Kaggle実行承認

- ユーザーの「実行してください」をtrain候補のcanonical採用とKaggle CPU pushの承認として反映した。
- compact self-contained trainをcanonical `*_train.ipynb`へJupytext生成し、17 cells、出力0、
  execution count 0を確認した。inferenceは未採用・disabledのまま。
- push対象は1 audit variant、LightGBM config 0、evaluation fold 5、trained fold 0、booster 0、
  HMM/PF well-run 0、control/parent再学習なし、private CPU、GPU/TPU/internet off。
- exp292 push前にhard prerequisiteのexp268 aggregate version 1をcanonical IDへpushした。
- exp268 aggregate version 1はkernel id `127887734`、約296秒でCOMPLETEした。773 wells /
  3,783,989 rows、zero-rate-spread 423 wells、prediction content SHA
  `fc18952f564dcefed8222ee30510828a4fb47f51c06a0eec5b1ddf37887ecdd1`を確認した。
- aggregate summary raw SHAは`8bd2064892f7eb05392785d602e810b9aea8b686225994cd515247609370e0c6`、
  manifest SHAは`427aa3f15c8577b38448836d3adea58ef69dcf43d6a79237f0c603b9bf04494b`。
  shard decompressed SHAとともにexp292 config/preflightへhard guardとして固定した。
- exp292 canonical packageはID `kentookumura/exp292-typewell-gr-warp-rate-identifiability-audit-train`、
  private CPU、GPU/TPU/internet off、kernel sources 5件。remote pull 403のため初回pushと判断した。
- package config SHA `d3bb0f040f77a3c057744b44665b7f6f7f207c882fc212a4186c32fdee74ca44`、
  train source SHA `4488aece9cf0dd998c4b25ba6a2d159d32db2b29686119e94a7cc3f41f31c5ab`。
  loose/package/bootstrap内config/sourceのbytes一致を確認した。
- 初回ID `kentookumura/exp292-typewell-gr-warp-rate-identifiability-audit-train`はID/titleの
  slug一致にもかかわらず`SaveKernel 400`となった。slugが56文字で、repo内の成功運用上限50文字を
  超えていたことが原因と推定した。失敗後pullも403でresource非作成を確認した。
- 同じexpのまま意味を保持した45文字のcanonical ID
  `kentookumura/exp292-typewell-gr-warp-identifiability-train` / title
  `exp292 typewell gr warp identifiability train`へ再prepareした。短縮IDも事前pull 403で未使用、
  metadataとloose config/source bytes一致を再確認した。
- 短縮canonical IDへのversion 1 pushは成功し、kernel id `127888550`。push後pullでprivate CPU、
  GPU/TPU/internet off、competition source 1件、kernel source 5件を確認した。

### 2026-07-19 Kaggle version 1完了

- `kentookumura/exp292-typewell-gr-warp-identifiability-train` version 1、kernel id `127888550`が
  `COMPLETE`。runtime 122.469秒、773 wells / 3,783,989 rowsを処理した。
- technical validationはPASS。事前登録した総合判定は`FAIL_CLOSE_NO_RESCUE_GRID`。
- primary H256 eligible coverageは29/773 wells = 3.7516%、row fraction 3.6178%で、両90% guardをFAIL。
- pooled candidate-best AUCはreal 0.484190、shuffled 0.531181、lift -0.046991。正のliftは0/5 folds。
- selected/safe pooled RMSEはともに11.938287、gain 0、改善0/5 folds。全773 wellsの選択はtail30 safe。
- 1000+、hidden-like spatial、hidden-like typewell-purgedは非悪化だが、safe fallback完全一致による。
- H256 real fallbackの主因はcommon finite pair不足219、tail30 forward-GR std不足159、common
  derivative pair不足152、tail30 derivative energy不足108、prefix Type Well GR std不足69、
  calibration slope範囲外36 wells。
- target-free scoreは23,190 records、decompressed content SHA
  `9165d52fb24152ea17c2a620247177ab4e4e223306869fba9bf5e9f59ca0ed01`、schema SHA
  `62181c0f628ef61c791918b1a3bb4813f36873f4679d3f72e21ba88adfdd7d9c`。
- target-free selectionは4,638 records、content SHA
  `5cebdca26a1a33f4d15fa141d465c7e2d187f1d42f555b9fc95edc61da03fdc1`、schema SHA
  `2be8f05d0fc878faf8534688c3e6f0bd8424183e12fe786bfd84a6dd6136bd8d`。
- fold manifest SHA `2c4d67f2d47cc44e215e16b7c4312631ed2c3ed51953b7d3667d3610b108493e`、
  summary raw SHA `4562c33b53bcf180874ef8962de618b56b50d4ce9e86688bb69d6c133522c557`。
- SHA manifestに記録された全artifactを取得物に対して再計算し、gzip decompressed SHAを含め不一致0。
- selected row prediction、model、inference、submissionは生成していない。固定停止条件どおり、
  rate/window/horizon/calibration/coverage/weight/thresholdの救済gridを行わずbranchを閉じる。
