# exp357 セッションノート

## 目的

旧exp344をreopenせず、exp342 patternから独立した固定Huber score監査として設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 1 actual exact-HMM完了、guard FAILで救済なし終了
- CV: Huber HMM `9.737195`、Stage 1 guard FAIL
- LB: なし

## 2026-07-23 設計

- exp357として採番し、steeringとscaffoldを作成した。
- Stage 0はHuber score 1 / saved Gaussian control 1 / 5 folds / HMM等0。
- Stage 1予約は1 variant / 773 HMM runs / control再実行0。
- delta 1.345と全gateを固定し、exp342をdependencyから外した。
- 実装、Notebook採用、Kaggle実行、inference、submissionは未実施。

## 2026-07-24 Stage 0実装

- ユーザーの「exp357を実装してください」をStage 0実装の承認として受領した。
- 正規`*_train.ipynb` / `*_inference.ipynb`は既存placeholderのまま保持し、
  compact self-contained Jupytext train候補とfail-closed inference候補を別名で追加した。
- fixed Huber `delta=1.345`を
  `0.5*z^2 (|z|<=delta)` / `delta*|z|-0.5*delta^2 (otherwise)`として実装した。
- exp280 Gaussian controlはdecompressed/content/scientific-contract SHAをhard guardし、
  再生成しない。
- exp226 safe列、512-row block、13 shifts、exp281 known-prefix sigma clip
  `[10,60]`、missing/typewell補間、foldを固定した。
- Huber/Gaussian bundleをtruth join前にcontent SHA固定し、同じstable SHA由来の
  nonzero circular rotationを両familyへ適用する。
- Stage 0固定gateはpooled MRR/top3各`+0.01`、各4/5 folds、long-tail、
  hidden-like 2面、persistent-offset、real-vs-circular、`|z|>=3` top3/regretのAND。
- Stage 1は予約だけで未実装。inference/submissionは明示停止する。

実行量契約:

- Stage 0 Huber scientific score: 1
- saved exp280 Gaussian control: 1（load only）
- shift candidates / reporting folds: `13 / 5`
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- parent/control再実行: 0
- Kaggle package/push/run、inference、submission: 0
- Stage 1予約: Stage 0全gate PASSと別承認時だけ1 variant / 773 HMM well-runs

実装・検証:

```bash
.venv/bin/python -m py_compile \
  experiments/exp357_exp226_huber_emission_independent_audit/*compact_selfcontained*.py \
  tests/test_exp357_exp226_huber_emission_independent_audit.py
.venv/bin/ruff check \
  experiments/exp357_exp226_huber_emission_independent_audit/*compact_selfcontained*.py \
  tests/test_exp357_exp226_huber_emission_independent_audit.py \
  --select F821,F401,F841,E722,E501
.venv/bin/pytest -q tests/test_exp357_exp226_huber_emission_independent_audit.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 \
  experiments/exp357_exp226_huber_emission_independent_audit/*compact_selfcontained*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp357_exp226_huber_emission_independent_audit/*compact_selfcontained*.py
make validate-exp EXP=exp357_exp226_huber_emission_independent_audit
```

- 専用test: `7 passed`
- exp280/exp342/exp357/Notebook関連test: `27 passed`
- py_compile / Ruff / Jupytext train・inference / strict validation: PASS
- compact train notebook: 21 cells（code 10 / markdown 11）、94,079 bytes
- compact inference notebook: 9 cells（code 4 / markdown 5）、7,894 bytes
- `__file__`、同一exp helper import、`from src`依存: 0
- config SHA:
  `1dc70d880b71dd7d82b70da4e578b8dd39530ab34364ef6294a339f8b0c7e75e`
- train source / notebook SHA:
  `f3a10bc8630b45f91c19385d153f5c5b5be2e61a1389df75a41b909abbc5f3fe` /
  `db25b0dc1ae7f1e9099078a5f29c8d578534dae81ee52219dd1d4fbd917f47cb`
- inference source / notebook SHA:
  `fed32a8d2bf0a4993b2eb93aef5ef915c07de02ef40f4f011be784b5c2ae7d6a` /
  `e5d54c1d9e0faf310499f2721239024739430379fff66cf32ec2426835a9463e`
- test SHA:
  `0a11d9e472e27e4f735b78420bd2d0099e5949ed85fc62b8c53e33cc214ff4f9`

親/参照構成比較:

- exp281正規self-contained trainは1,526行/10章でfull exact-HMM生成を扱う。
- exp342 compact trainは2,942行/12章でStage 0とStage 1を扱う。
- exp357 compact trainは1,669行/10章。exp342 Stage 0のruntime/input/
  target-free scoring/truth-late readout/gate/orchestrationを維持し、
  Student-tとStage 1 HMMを除いてfixed Huberだけに限定した。

リポジトリ全体test:

```bash
make test
```

- 実装時点: `827 passed, 6 skipped, 3 failed`
- exp357専用7件は全体実行でもPASS。
- 失敗2件は既存`tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py`。
  完了後config status `completed_train_side_guard_failed_closed`に対してtestが
  `kaggle_cpu_*` prefixを要求し、`execution.run_variant=false`に対して旧approval
  guard順を要求する既知不一致。
- 残る1件は既存
  `tests/test_exp375_exp362_prefix_rate_fixed13_dual_selector.py`。
  実行済みconfigの`execution.approval_consumed=true`に対してtestが旧値`false`を
  要求する不一致。
- 上記3件の対象config/testは今回変更しておらず、exp357実装のためには修正しない。

## 再現性メモ

- real scoreはRNGなし。circular controlはcontent SHAから固定する。
- exp280/281/226 input SHA、Huber score、block readout、gate SHAを記録する。
- Stage 1時だけdecoder/prediction SHAを保存する。
- deterministic anchorとは扱わない。

## 次のアクション

固定Huber Stage 1はtail-safety / direct ceilingをFAILしたため、
救済、再実行、inference、submissionへ進まない。

## 2026-07-24 Kaggle Stage 0実行承認

- ユーザーの「実行してください」を、compact self-contained train候補の正規Notebook採用、
  Kaggle package/push、private CPU Stage 0を1回実行する承認として受領した。
- scientific score: fixed Huber `delta=1.345` 1件
- saved control: exp280 Gaussian 1件（SHA固定load-only、再生成0）
- shift candidates / reporting folds: `13 / 5`
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- parent/control再学習: 0
- runtime: Kaggle CPU、GPU/TPU/internet off
- 未承認: Stage 1の1 variant / 773 HMM runs、inference、submission
- canonical kernel:
  `kentookumura/exp357-huber-emission-independent-audit-train`
- canonical title: `exp357 huber emission independent audit train`

正規採用・package検証:

- 正規train notebookはcompact候補とcell source 21/21一致。
- 正規train notebook: 21 cells（code 10 / markdown 11）、
  SHA `eb8f3d5a119d644d2417423201ae7d89a31ffb093c88bec96227cd08943ba4fb`
- 専用＋Notebook tests: `11 passed`
- py_compile / Ruff / Jupytext round-trip / strict experiment・project validation: PASS
- canonical slug検索: `Not found`。新規canonical kernelとしてpushする。
- package metadata: private、CPU、GPU/TPU/internet off、run-on-push
- competition source: 1、kernel sources: exp226 / exp280 / exp115の3件
- package notebook: 22 cells（bootstrap code 1 + 正規21）、
  SHA `dd8841f22049affc35a3f3984d3c3d31dcbd8ada1fe2042dce4a55f2dccf30e6`
- metadata SHA:
  `6aac8ee01709065e080f85a9da489d9cf55f8dbfca00e6690fd4134b474376b2`
- config loose/package/bootstrap SHA:
  `31ac2c12e0d23e8e439757b8be5eb155724c5efe512ccb0a0f239d0365f6c2b5`
  で一致。
- bootstrap support files: 22、ZIP SHA:
  `1dc3b725f21efebc004fc4c9cce43143386e2c692b15f3f8cf40232186ea5bd6`
- bootstrap内train source SHA:
  `f3a10bc8630b45f91c19385d153f5c5b5be2e61a1389df75a41b909abbc5f3fe`
  で正規sourceと一致。

push・canonical照合:

- canonical private CPU kernel version 1をpushした。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp357-huber-emission-independent-audit-train`
- canonical id_no: `128448451`
- pull metadata: id/title、private、CPU、GPU/TPU/internet off、
  competition source 1、kernel sources 3がpackageと一致。
- remote notebook: 22 cells、local packageとcell source 22/22一致。
- 初期status: `KernelWorkerStatus.RUNNING`
- 重複実行防止のため、push直後にroot configの`run_stage_0=false`、
  `train_run_on_push=false`へ戻した。Kaggleへ送信済みversion 1のbootstrap configは
  SHA `31ac2c12...6c2b5`の承認済みtrueを保持する。

## 2026-07-24 Kaggle Stage 0結果

- canonical private CPU version 1（id_no `128448451`）は
  `KernelWorkerStatus.COMPLETE`。
- diagnostic runtime: `319.61734890937805 sec`
- rows / wells / blocks: `3,783,989 / 773 / 7,787`
- scientific score / saved control / shift / reporting fold:
  `1 / 1 / 13 / 5`
- HMM well-run / model config / trained fold / booster:
  `0 / 0 / 0 / 0`
- parent/control再学習、inference、submission: すべて0

固定gate結果:

- technical control:
  score finite / row identity / saved Gaussian rank parityは各`1.0`でPASS。
- pooled MRR:
  Huber `0.3896675435`、Gaussian `0.3896259848`、
  gain `+0.0000415588 < +0.01`でFAIL。
- pooled top3:
  Huber `0.4525491203`、Gaussian `0.4524207012`、
  gain `+0.0001284192 < +0.01`でFAIL。
- improved folds: MRR / top3とも`2/5 < 4/5`でFAIL。
- stress MRR / top3 non-regression: 両方FAIL。
- real-vs-circular gap non-regression: MRR / top3ともPASS。
- extreme residual 174 blocks:
  top3 gain `+0.0114942529`、regret delta `-0.6522697228 ft`で両方PASS。
- Huber top1 margin平均`0.0172596212`はGaussian `0.0183269273`より小さく、
  `flattening_signal=true`。

判定:

- `decision=stage_0_failed_close_without_rescue`
- `stage_1_eligible=false`
- Stage 1の1 variant / 773 HMM runsは未実装・未実行。
- delta/scale/sigma/tempering/blend救済、再実行、inference、submissionは行わない。

主要SHA:

- target-free score content:
  `832552dded42940ac57cc9aac425d1ea1be7b0d6b6e17950d792f7e1c9a95902`
- target-free score decompressed:
  `65403a1c56e666782a9c7e423d2b8c42b27977fb43d24c57cdda5b1750f4029a`
- block readout content:
  `7a296bf21ada0c3366007cbeb77d9f231679e4ad36366cb218d85cf06a6c6276`
- block readout decompressed:
  `4e6870fbc4e597a1cfc677a768cdd46760b9207fce7c0d61a5715d336c4f231a`
- gate:
  `b79190de4e4d737a71f37373e2efd73a86b8e68712db952e21d245eb418e4dd7`

AGENTS方針どおり、CV/gate/variant/SHAはKaggle logsで十分確認できたため、
output archive全体はダウンロードしなかった。

完了後検証:

- exp357専用＋Notebook tests: `11 passed`
- py_compile / Ruff / strict experiment validation / project validation: PASS
- 全体test: `828 passed, 6 skipped, 2 failed`
- 残る2件は今回未変更の既存exp296 config完了状態と旧test期待の不一致。
  exp357専用testは全体実行でも7/7 PASSした。

## 2026-07-24 Stage 1明示override

- ユーザーの「HMM実行に進んでください」を、Stage 0 FAILの通常停止条件を
  明示的にoverrideし、同じexp357内でfixed Huber Stage 1を1回実装・実行する
  承認として受領した。
- scientific variant: `huber_delta1p345_residual_offset_delta80_step035_rate41` 1件
- HMM well-runs / reporting folds: `773 / 5`
- model config / trained fold / booster: `0 / 0 / 0`
- parent exp281 Gaussian HMM再実行: 0。SHA固定saved OOFをload-onlyで比較する。
- Stage 0再実行、GPU/TPU、inference、submission: すべて0
- runtime: Kaggle CPU、internet off、Numba threads 4
- fixed parent contract:
  offset `[-80,80] ft`、step `0.35`、rate states 41、`sig_r=0.002`、
  `sig_p=0.02`、`lam=1.0`、`mom=0.998`、known-prefix sigma、
  exp226 shape/transition、posterior mean。
- sole intervention:
  exp281 Gaussian行別emissionをHuber `delta=1.345`へ置換。
- Stage 1 gate:
  exp281比`>=0.05 ft`、改善4/5 folds、1000+・hidden-like 2面非悪化、
  by-well p95非悪化、worst `<=+0.25 ft`、parent/exp226 parity、
  finite/row identity 1.0、direct RMSE `<=9.427109596582213`。
- Stage 0を再計算せず、Stage 1 candidate全well pathをtruthなしでfreezeしてから
  saved exp281 OOFとtruthをjoinする。
- exp342 Stage 1 exact-HMM kernelを構成参照元とし、遷移核は数値parity testで
  exp281 `_hmm2_fb`と一致させた。変更はHuber row likelihoodと列名だけ。
- 実装後の専用test: `10 passed`
- py_compile / Ruff / strict experiment validation: PASS

Stage 1 canonical採用・package監査:

- exp342 compact train 2,942行 / 12章に対し、exp357は2,943行 / 12章。
  Stage 0のHuber監査、Stage 1 exact-HMM、saved-parent評価、setup、entrypointを
  同じ粒度で展開した。
- canonical train notebook: 25 cells。packageはbootstrap 1 + canonical 25 = 26 cells。
- package/canonical cell source parity: `25/25`
- exp357専用＋Notebook tests: `14 passed`
- Jupytext train/inference round-trip、py_compile、Ruff、strict experiment、
  project validation: PASS
- config loose/package/bootstrap SHA:
  `9804f1ee48ae552737430c7c3610592059e7b3a5fb097d386a14cf0782bafcf6`
- train source loose/bootstrap SHA:
  `a6c27a2873809f79a4a67a29595974012642e00035d102d09f9305413b9c613e`
- canonical notebook SHA:
  `1bbf975f5fc92830033358631c78a88c84bf4d29753c2e78c25614999cf2d7de`
- package notebook SHA:
  `016c6736c89ba17fa2cc77ff97a03e38c7caa45f9d265c24932874a15949f633`
- kernel metadata SHA:
  `040380f4437bedb9b4e57b5ef6853736ce84a16c000698ae38d71b52c09a1435`
- bootstrap support files: 22、ZIP SHA:
  `7cafd0671f6786557acf202c11bac84a910c19189fbebb9111dcbb4316576480`
- package metadata: same canonical id/title、private CPU、GPU/TPU/internet off、
  run-on-push true。
- competition source: 1、kernel sources:
  exp226 / exp281 / exp115の3件。exp280 Stage 0 sourceはversion 2で不要。
- push前に既存canonical version 1をpullし、id_no `128448451`、
  private CPU/internet off、同じslugの存在を再確認した。

Stage 1 push・remote照合:

- same canonical private CPU kernel version 2をpushした。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp357-huber-emission-independent-audit-train`
- canonical id_no: `128448451`
- remote metadata: id/title、private、CPU、GPU/TPU/internet off、
  competition source 1、kernel sources exp226 / exp281 / exp115がpackageと一致。
- remote notebook: 26 cells、local packageとcell source 26/26一致。
- initial status: `KernelWorkerStatus.RUNNING`
- 重複実行防止のためpush直後にroot configの`run_stage_1=false`、
  `kaggle_push_approved=false`、`train_run_on_push=false`へ戻した。
  Kaggleへ送信済みversion 2のbootstrap configはSHA
  `9804f1ee...afcf6`の承認済みtrueを保持する。

## 2026-07-24 Kaggle Stage 1結果

- canonical private CPU version 2（id_no `128448451`）は
  `KernelWorkerStatus.COMPLETE`。
- created at: `2026-07-24T10:36:05.844586+00:00`
- exact-HMM runtime: `9597.242200136185 sec`
- rows / wells: `3,783,989 / 773`
- scientific variant / HMM well-run / reporting fold: `1 / 773 / 5`
- model config / trained fold / booster: `0 / 0 / 0`
- parent Gaussian HMM再実行、GPU、inference、submission: すべて0
- execution basis:
  `explicit_user_override_after_stage_0_fail`

実HMM結果:

- exp281 saved Gaussian RMSE: `9.827419940583813`
- fixed Huber `delta=1.345` RMSE: `9.737195157482754`
- gain vs exp281: `+0.09022478310105875 ft`で必要`>=0.05 ft`をPASS。
- 改善fold: `4/5`でPASS。fold 0だけ
  `7.9879271460 -> 8.0027389254`と`+0.0148117794 ft`悪化し、
  fold 1--4は改善した。
- long-tail 1000+:
  `10.8347427249 -> 10.7469746621`、`-0.0877680628 ft`でPASS。
- hidden-like spatial:
  `10.5566073441 -> 10.3479808797`、`-0.2086264644 ft`でPASS。
- hidden-like typewell-purged:
  `10.3041393322 -> 10.3028943381`、`-0.0012449941 ft`でPASS。
- by-well p95 delta:
  `+0.0033652907 ft > 0`でFAIL。
- worst well `4a8ecc0b`:
  `+1.4037152572 ft > +0.25 ft`でFAIL。
- direct exp226 ceiling:
  candidate `9.7371951575 > 9.4271095966`、
  `+0.3100855609 ft`でFAIL。
- finite / row identity / parent parity / exp226 parity: PASS。

判定:

- `decision=stage_1_failed_close_without_rescue`
- `promotion_guard.passed=false`
- `direct_promotion_passed=false`
- Stage 0 proxy FAILに反してactual HMMは平均、4/5 folds、required scopeを改善した。
  ただしby-well tailとdirect ceilingが不足するため採用しない。
- delta/scale/sigma/tempering/blend救済、再実行、inference、submissionは行わない。

主要SHA:

- candidate content:
  `784885201e8faf5ecb9d4a91d8722d3477572b2b9e837fb5d76cf72e0b50c4a6`
- prediction raw gzip:
  `3560c7d069ddf27ed81b8eb22fba06a8ac9de3db1d27166861563bcf36841cad`
- prediction decompressed:
  `1a8e94c2d54227c1afe49e89982d97f636fbc1bf814500765a0efb4cf26221b8`
- decoder manifest mapping / file:
  `0076116b30a8712dbba9f18bb05d17fd870957aa70014f2094be61c19d057847` /
  `a090784f1c9f9b69263dadea852ff4a0ad01151f13c656e6ebab1dcbba2fe236`
- Stage 1 gate:
  `52b6f23e3d46e99ddc5940653edb80f723eb356c7022d313b116ec7700249829`

CV、fold、scope、by-well guard、SHAはKaggle logsで確認できたため、
AGENTS方針どおりoutput archive全体はダウンロードしなかった。
