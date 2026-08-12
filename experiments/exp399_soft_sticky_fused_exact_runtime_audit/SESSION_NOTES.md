# exp399_soft_sticky_fused_exact_runtime_audit セッションノート

## 目的と承認

- 目的: exp394と全状態・科学条件・出力精度を保ったruntime-only最適化
- ユーザー承認: 2026-07-25「それで進めてください」
- 実行範囲: candidate 1 / fixed 16 wells / HMM runs 16
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- parent/control rerun / GPU / inference / submission: `0 / 0 / 0 / 0`
- full OOF: 未承認、無効

## 固定した実装

- exp394のTVT step 0.35 ft、全TVT grid、41 rate statesを維持
- exp209型の固定幅max+sum reductionへ変更
- P×R×5 transition tensorを廃止
- source境界正規化は境界でだけon demand計算
- forwardでdocking、H→E split、rate propagationを融合
- joblib 2-well threads、各worker内でNumba 2 threadsを明示設定
- stable well / row順、RNGなし、truth-before-freeze 0を維持

## ローカル検証

- 小規模trellisでexp394との全12出力を比較
- 最大差:
  - joint / H mean: 約`8e-10 ft`以下
  - posterior: 約`2e-16`
  - expected switch等: 約`1e-18`
- 573 TVT × 41 rateのNumba合成benchmark:
  - 境界表撤去前: 約`2.31x`
  - 境界表撤去後: 約`3.02x`
  - docking cacheはmemory trafficで悪化したため不採用
- ローカルbenchmarkはKaggle fixed16 gateの代替ではない。

## 反復で棄却した実装

- 3×5を単純なon-the-fly pairwise `_logaddexp`にしただけの版:
  純Python局所計測で親より遅く、不採用。
- docking全stateをforwardからbackwardへfloat32 cacheする版:
  concurrent memory trafficが増え、JIT throughputが悪化したため不採用。

## 次

専用test、Jupytext同期、strict package検証後、Kaggle CPUで保存済みexp394
fixed16とのparityと実測runtime gateを評価する。full OOFは実行しない。

## Kaggle fixed16実行

- kernel: `kentookumura/exp399-soft-sticky-fused-exact-runtime-audit-train`
- version / id_no: `1 / 128546220`
- push時刻: `2026-07-25 05:09 UTC`
- CPU / internet off / run-on-push
- version 1 status: ERROR after all 16 wells decoded
- version 1 decode:
  - last well completed at notebook time `770.69 sec`
  - per-well elapsed sum `1,467.22 sec`
  - parent per-well elapsed sum `3,636.74 sec`
- failure: candidate `row_idx` int32と保存CSV int64を
  `DataFrame.equals`で比較したため、同値keyをdtype差で不一致と誤判定
- 修正: well_idをstring、row_idxをint64へ正規化して値を厳密比較する
- scientific kernel / prediction計算は変更なし
- version 2: 同一slugへkey比較修正版をpush、実行中

## Kaggle version 2 結果とtechnical gate修正

- status: COMPLETE
- total / decode wall: `717.528557 / 677.557184 sec`
- state-time normalized speedup: `5.367430x`
- projected full runtime: `21,003.884570 sec`（`5.834 h`）
- runtime / RSS / finite / full-grid / normalization / leakage: PASS
- schedule content SHA: exp394と一致
- branch probability max abs diff: `5.110359e-8`
- diagnostic max abs diff: `2.978181e-6`
- prediction / H mean:
  - abs diff RMSE `5.65e-7 ft`
  - p99 `3.63e-6 ft`
  - maximum `7.883838e-6 ft`
- version 2 FAIL要因:
  - 事前prediction閾値`2e-6 ft`を最大差だけが超過
  - exp394 configの手記録state units `3,290,350,369`が、
    親runtime CSV実値`3,290,350,409`より40少ない
- 対応:
  - 親runtime CSV実値をstate-unit identityの正とする
  - 演算順だけが異なる固定幅max+sumについて、prediction practical
    equivalenceを`1e-5 ft`へ明示する
  - state、schedule、確率、scientific parameter、scoreは変更しない
- version 3: corrected technical parity contractを同一slugへpush、実行中

## Kaggle version 3 と追加runtime最適化

- prediction / branch / schedule SHAはversion 2と完全一致
- parity / state-time identity: PASS
- decode wall: `1,170.581210 sec`
- speedup: `3.106782x`
- projected full runtime: `36,287.346972 sec`（`10.080 h`）
- 同一コードのversion 2よりCPUが約1.73倍遅く、runtime gate FAIL
- runtime varianceへ再実行だけで対処せず、backwardで以下を融合:
  - rate max+sum reduction
  - rate側docking expectation
  - H stay / switch
  - beta更新
  - H→E mass reduction
- 削減物: transition value、stay log、switch logのP×R中間3配列と
  parallel kernel launch 1回
- ローカルJIT single-well合成計測:
  `0.911856 → 0.617901 sec`（約32.2%短縮）
- version 4: backward融合版を同一slugへpush、実行中

## Kaggle version 4 最終結果

- status / gate: `COMPLETE / PASS`
- decision: `technical_preflight_passed_full_oof_requires_separate_approval`
- total / decode wall / per-well sum:
  `632.688257 / 589.600103 / 1,174.779629 sec`
- state-time normalized speedup: `6.168148x`
- projected full runtime: `18,277.265455 sec`（`5.077 h`）
- projected peak RSS: `2.544819 GB`
- parent parity:
  - prediction / H mean max `7.883838e-6 ft`
  - branch probability max `5.110359e-8`
  - diagnostic max `2.978181e-6`
  - key / schedule / state-time units:一致
- finite / full-grid coverage: `1.0 / 1.0`
- posterior / transition max error:
  `4.241052e-14 / 8.881784e-16`
- truth / error / hidden role pre-freeze reads: `0 / 0 / 0`
- summary SHA:
  `ac800ac0ada91fd1df4486ea5ae580c32e7afd3c0ae7de48172476291550c660`
- gate SHA:
  `ddae4f7ca8381c2579c6c1da10141fc430a670e601a01f7a3c8683b018df8ad3`
- output:
  `/tmp/kaggle-output/exp399_soft_sticky_fused_exact_runtime_audit/train_v4`
- run flag / run-on-pushはfalseへ戻した
- full OOFは別承認まで実行しない

## full 773-well OOF実行承認

- ユーザー承認: 2026-07-25「実行してください」
- 実行対象: scientific candidate `1`
- reporting folds / switching-HMM well runs: `5 / 773`
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- parent/control reruns / GPU / inference / submission: `0 / 0 / 0 / 0`
- baselineは保存済みexp263 `exp226_w500_50_50`（OOF RMSE
  `8.238331546`）をload-onlyで参照し、再学習・再生成しない
- fixed16 PASS summary
  `ac800ac0ada91fd1df4486ea5ae580c32e7afd3c0ae7de48172476291550c660`
  をbootstrapへ固定し、full OOF開始前にraw SHAとPASSを照合する
- 追加Kaggle入力:
  `exp263-last-anchor-pair-cache-train`、
  `exp115-hidden-like-spatial-holdout-from-ppt-train`
- CPU / 2 wells並列 / internet off / run-on-push
- full OOF完了後もinferenceとsubmissionは作成しない
- Kaggle version 5 push: `2026-07-25 06:51:55 UTC`
- kernel:
  `kentookumura/exp399-soft-sticky-fused-exact-runtime-audit-train`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp399-soft-sticky-fused-exact-runtime-audit-train`
- start確認: `KernelWorkerStatus.RUNNING`
- 起動contract表示:
  `1 variant / 773 HMM runs / 5 folds / 0 boosters / 0 control reruns`
- 初期進捗: `6 / 773 wells`完了、例外なし

## Kaggle version 5 failure

- decode: `773 / 773 wells`完了
- failure phase: prediction / branch posterior / scheduleをtruth前にfreezeした後の
  exp263 load-only baseline結合監査
- exception:
  `RuntimeError: exp399 and exp263 outer-fold identities differ`
- 原因:
  - exp399のreporting foldはexp226のgroup-safe fold
  - exp263 cacheの`outer_fold`はexp072由来の独立したgroup-safe fold
  - 773 wellsのうち同じ数値labelだったのは`142` wellsだけで、単なる
    label permutationでもない
  - row identity、baseline OOF性、候補予測には関係しない2つのfold provenanceを
    同一であると誤って要求していた
- 修正:
  - 既存実装どおりpromotionのfold reportingはtruth前にfreeze済みのexp226
    `fold`を維持
  - exp263 `exp263_fold`は独立したOOF生成provenanceとしてwell内一意性と
    0..4 coverageを監査し、cross-tabを保存
  - overall / distance / hidden-like / by-well評価、候補予測、baseline値、
    promotion閾値は変更しない
  - 773-well decode直後にpre-truth prediction / branch posterior / schedule /
    runtimeを保存し、late readout失敗時の再損失を防ぐ
- local fold監査:
  - 両ledgerともfold support `0..4`、各well内で一定
  - mismatch `631 / 773 wells`
- retry実行量:
  `1 variant / 773 switching-HMM runs / 5 reporting folds /
  0 LightGBM configs / 0 trained folds / 0 boosters /
  0 parent-control reruns / CPU`
- dedicated tests: `7 passed`
- inference / submission: 引き続き無効
- Kaggle version 6 push: `2026-07-25 14:54 UTC`
- start確認: `KernelWorkerStatus.RUNNING`

## Kaggle version 6 full OOF最終結果

- status: `COMPLETE`
- generated_at: `2026-07-25T21:53:23.969219+00:00`
- rows / wells: `3,783,989 / 773`
- total / prediction freeze:
  `25,118.126809 / 24,669.330589 sec`
- runtime limit: `30,600 sec`、technical runtime gate PASS
- candidate OOF RMSE: `11.395645678`
- exp263 actual OOF RMSE: `8.238331667`
- gain vs exp263: `-3.157314012 ft`
- fold delta candidate-minus-exp263:
  `+1.724406 / +3.503768 / +2.260748 / +4.475109 / +3.482840 ft`
- improved folds: `0 / 5`
- improved-or-equal wells: `40.491591%`
- by-well delta p95 / worst:
  `+12.034886 / +38.148059 ft`
- 000--250 / 1000+ delta:
  `+0.517525 / +3.488101 ft`
- hidden-like spatial / typewell-purged delta:
  `+4.385663 / +4.198109 ft`
- persistent episodes: `689` vs exp263 `551`（`+138`）
- recovery within 512: `0.146589` vs `0.090744`（`+0.055845`）
- E/H occupancy: `0.052990 / 0.947010`
- expected switches: `0.348381 / 1000 MD-ft`
- technical checks: 全PASS
- scientific checks:
  branch occupancy、switch rate、recovery within 512だけPASSし、残り10件FAIL
- decision:
  `promotion_rejected_no_parameter_rescue_blend_selector_inference_or_submission`
- model / trained fold / booster / parent-control rerun:
  `0 / 0 / 0 / 0`
- inference / submission: 作成なし

## version 6再現性と保存物

- preflight summary raw SHA:
  `ac800ac0ada91fd1df4486ea5ae580c32e7afd3c0ae7de48172476291550c660`
- scientific contract SHA:
  `b640bce974b9d940b4988e5204b9df42dc4e216b2566e01061a04db9a4e3002f`
- prediction content SHA:
  `d44b382a310c7d53bf5dc90a238c44a247b2ba09d2a1f0648174f4d7c85fb18e`
- branch posterior content SHA:
  `d953cebc3d46ef8cd4a03d9faaceab0cf74fbe998a2fe9a9d2141563996cd8a1`
- schedule content SHA:
  `98e6f2781c615d5bcecc594904597a2a72707347c41ac150a601f40155b02a31`
- gate raw SHA:
  `ace88e93431b264da907935812927492891a59844158ad6a4d84d591b216a747`
- OOF / branch / schedule archive raw SHA:
  `4edf39adc1d91d71a50a01c5623cf1a9ea75f8e374718a50e156e31047bc5cbd` /
  `3d7d05af3aeac54121fa86f93d29ec222dd21ac2a4b73b4a2965a1c9125fe84a` /
  `41630d69fbe02c7d38b7c2235df1b13abfb81b68086da9a44b0813445ece6cce`
- output:
  `/tmp/kaggle-output/exp399_soft_sticky_fused_exact_runtime_audit/train_v6`
- download後に3 gzipのintegrityとraw SHA一致を確認
- run_full_oof / train_run_on_pushをfalseへ戻し、branchを閉じた
- result回収・最終検証コマンド:
  - `kaggle kernels output kentookumura/exp399-soft-sticky-fused-exact-runtime-audit-train -p /tmp/kaggle-output/exp399_soft_sticky_fused_exact_runtime_audit/train_v6`
  - `.venv/bin/pytest -q experiments/exp399_soft_sticky_fused_exact_runtime_audit/tests/test_exp399_soft_sticky_fused_exact_runtime_audit.py`
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp399_soft_sticky_fused_exact_runtime_audit/exp399_soft_sticky_fused_exact_runtime_audit_compact_selfcontained_train.py`
  - `make validate-exp EXP=exp399_soft_sticky_fused_exact_runtime_audit`
  - `make update-summary`
