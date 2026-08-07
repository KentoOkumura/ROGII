# exp392_exp389_fixed13_dual_selector_on_exp264 セッションノート

## 目的

exp389の固定`delta=1.345` Huber absolute-TVT exact-HMM train予測を、
corrected exp264 fixed12 candidate-long dual selectorへ13本目として追加する。
単体平均改善をselectorが安全に利用できるかを評価し、exp389のtail gate失敗自体は
PASSへ再分類しない。

## 事前固定設計

- Route: `ensemble`
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp389_exp209_huber_exact_hmm_emission`
- 追加候補: `huber_exact_hmm`
- fixed fallback: 既存7候補のまま
- selector fold: exp263 outer 5 fold
- active variant: `1`
- LightGBM objectives: `2`
- outer / inner folds: `5 / 4`
- planned CPU selector boosters: `40`
- parent/control retraining: `0`
- GPU boosters: `0`
- downstream TVT / inference / submission: `0 / 0 / 0`

ユーザーの「平均で改善しているのなら次に進んでください。selectorの候補に
入れるのが次です。」を、上記Stage A + Stage C CPU実行までの承認として記録した。
科学gateをFAILしたexp388 Student-t候補は併用せず、fixed14にはしない。

## exp389入力契約

- kernel: `kentookumura/exp389-exp209-huber-exact-hmm-emission-train`
- version / id_no: `1 / 128466838`
- file:
  `artifacts/exp389_exp209_huber_exact_hmm_emission_predictions.csv.gz`
- rows / wells: `3,783,989 / 773`
- raw gzip SHA:
  `95302d547e8c49cdf67dabe6200e08e5c83f01ea158cf2fbd4f25b2fd1f74d75`
- decompressed/logical SHA:
  `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
- allowlist:
  `id,well_id,row_idx,huber_delta1p345_on_exp209_absolute_tvt_hmm_tvt,
  huber_delta1p345_on_exp209_absolute_tvt_hmm_std,
  huber_delta1p345_on_exp209_absolute_tvt_hmm_loglik`

exp389はwell間学習を持たず、各wellのknown prefixだけで生成したtarget-free候補。
source foldは存在しないため合成せず、global `(well_id,row_idx)` join後に
exp263 selector foldへpartitionする。truth、error、Gaussian/LikPF control、
scope、gate診断はloaderで開かない。

## 科学gate

- selector score guardをPASS
- added candidate primary top1 fraction `>= 0.005`
- positive usage folds `>= 4`
- parent fixed12 hard selectorに対しpooled非悪化
- parent fixed12に対し`4/5` folds改善
- near 0--250 / 1000+ / hidden-like delta `<= 0.02 ft`
- by-well p95 / worst delta `<= 0.25 ft`
- same-OOF rescue、weight、threshold調整は禁止

## 実装

- steering:
  `.steering/20260725-exp392-exp389-fixed13-dual-selector-on-exp264/`
- experiment:
  `experiments/exp392_exp389_fixed13_dual_selector_on_exp264/`
- reusable loader/cache:
  `src/exp389_fixed13_candidate_cache.py`
- dedicated tests:
  `tests/test_exp392_exp389_huber_fixed13_dual_selector.py`
- Jupytext source:
  `exp392_exp389_fixed13_dual_selector_on_exp264_compact_selfcontained_train.py`

exp388のfixed13 selector構成を参照し、candidate sourceだけをexp389の実列契約へ
置き換える。Notebook内で入力/SHA、cost contract、global key join、Stage A、
Stage C、paired readout、novelty、生成物を追跡する。同一exp helper importと
`__file__`は使用しない。

## 再現性

- global seed: `42`
- LightGBM: `deterministic=true`、`force_col_wise=true`
- stable SHA256 sampling keyをfold/objective/candidateから作る
- exp389 gzip raw SHAとdecompressed/logical SHAをhard checkする
- feature schema/content、40-model manifest、candidate score、summaryのSHAを記録する
- Kaggle private CPU、GPU/internet off
- 保存済み親artifactだけを使い、親controlは再学習しない

## 現在状態

`completed_fixed13_selector_scientific_gate_failed_closed`

## push前検証

- Jupytext `--test`: train / inference PASS
- py_compile / Ruff `F821,F401,F841,E501`: PASS
- dedicated + common selector tests: `55 passed`
- strict experiment / project validation: PASS
- package configと正本config: push時点でbyte一致
- metadata: private、CPU、GPU/internet off、run-on-push
- kernel sources:
  exp263 cache / exp389 Huber prediction / exp264 parent score
- 事前確認:
  `1 variant / 2 objectives / outer 5 / inner 4 /
  40 CPU selector boosters / parent-control retraining 0`

push時のpackage SHA:

- config:
  `761a214f3f7643e2f6306b4a5ae29ad6acf8193d2080dd8cab5ea7e1eb5705ae`
- metadata:
  `d8df8a3cc4dac2ac537856ede376d2f4f95345d6e09e97d5ee1d140d805b6d82`
- packaged notebook:
  `dc25c2e7a5829fd0f1de9575748fbeffa9011d74d68d8067c9845ec40c1b50ed`
- candidate contract:
  `1538eb86de4964bd6cbbaeae4a4bf237607934e06c55858227adad038b0193ae`
- feature contract:
  `f888f5023effde154adeb61c85176e9f25502c5938f2f581b4e0c21943ae7a09`
- exp389 fixed13 helper:
  `adfeb3a225daf6a4e8b766ce8508656e36cebc838603993e6fd064adc145586b`

## Kaggle push

push前のcanonical slug pullは新規Notebookのため403となり、同slugの既存versionが
ないことを確認した。2026-07-24 23:55:33 UTCにversion 1をpushし、同slugを
pullできることとRUNNINGを確認した。

- kernel:
  `kentookumura/exp392-exp389-huber-fixed13-selector-train`
- version / id_no: `1 / 128523057`
- status: `KernelWorkerStatus.COMPLETE`
- approval: version 1で消費済み
- pulled metadata: GPU off / internet off / machine shape `None`

空logsや一時的なstatus障害を理由に再pushせず、同じversion 1を完了まで監視した。

## Kaggle version 1結果

- notebook scientific runtime: `3666.541645113 sec`
- selector models: `40 / 40`
- parent/control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / 0 / 0 / 0`
- technical checks / leakage audit: 全PASS
- exp389 global-key join / selector fold repartition: PASS
- exp389 truth/error loaded: `0`
- exp389 native confidence finite率: `1.0`
- Stage A: 650,000 audit rows、153 -> 90 features、compact 77
- Stage C: 25 partitions、18,919,945 compact rows、
  49,191,857 outer-valid candidate-score rows

selector score guardはPASSした。

- expected-error MAE: `5.854091105 -> 3.845602339`
- within10 logloss: `0.510918637 -> 0.359695378`
- within10 Brier: `0.165363978 -> 0.111966344`
- 3指標すべてpooled・5/5 folds改善

## Fixed13 integration

- fixed13 hard RMSE: `8.769791682`
- parent fixed12 hard RMSE: `8.652531956`
- delta: `+0.117259726 ft`悪化
- fixed fallback RMSE: `8.238331546`
- fixed fallback error parity max abs: `0.0 ft`
- fold delta:
  `-0.035899949 / +0.243168613 / +0.385042054 /
  -0.131226503 / +0.111191785 ft`
- improved folds: `2 / 5`
- Huber top1: `91,035 / 3,783,989 rows = 0.024057945`
- positive usage folds: `5 / 5`
- near 0--250 delta: `+0.017905454 ft`（PASS）
- 1000+ delta: `+0.126035372 ft`（FAIL）
- hidden-like spatial / typewell-purged:
  `+0.160511723 / +0.154788720 ft`（FAIL）
- improved / regressed wells: `343 / 430`
- by-well median / p95: `+0.014664501 / +0.774302299 ft`
- worst `8902c3f6`: `+7.875187526 ft`

利用率、near、selector scoreだけをPASSし、pooled、改善fold数、1000+、
hidden-like 2面、by-well p95、worst-wellをFAILした。decisionは
`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`。

## Post-freeze novelty

診断専用oracleの補完性も小さかった。

- H512: `3.700319996 -> 3.696656821`、`0.003663175 ft`改善、
  unique-best `275 / 7,787 groups`
- whole-well: `4.801786361 -> 4.791666110`、`0.010120251 ft`改善、
  unique-best `29 / 773 wells`

## Post-hoc reranking診断

- worst `8902c3f6`のHuber top1率: `0.0`
- usage-delta Pearson / Spearman:
  `0.004539202 / -0.010965210`
- Huber利用0 wells: `285`
- そのうち悪化: `158`

追加候補を直接選ばないwellでも大きく悪化しており、Huberの直接誤選択だけでなく、
selector再学習による既存12候補のreranking不安定性を支持する。このpost-hoc診断は
科学gateには使っていない。

## 再現性SHA

- exp389 decompressed:
  `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
- post-read prediction:
  `b16e91d3493f168b6d4a527d157febb9940120d80e729eb09d657f8d5d9445ad`
- feature schema:
  `9fa5f2373a7fbfa566f880b1985a3f0f1689807ba36c33299e35f50d2e236baa`
- model manifest:
  `e9b03df33755f1be15bd11e76254d0b3592144202d8de1c39b670ca9dcf5b625`
- compact manifest:
  `9a818679ef21f3f3481590d5448a8339cc9a9c58bc08fc0e9a2d014a667f9bf0`
- outer-valid candidate score:
  `b4de2552ef4806f2ee644c201669e14ad78ca4e47f2df922029dded27e5472a0`
- scope / usage / by-well / gate:
  `f90e01c33d4f79c0e01ed57d6991bb462cf5889bee053b75d9c7810c21baaa96` /
  `b8960a909b649824a91855a61aac2e31181ccdd6cfcc2b1c8c3c4c7b09308ec5` /
  `73bab27f8907dccdefcc182862688fd70a12e3670d44d2173b326745ca98d9f0` /
  `90bfb3f4d0a3e277ea1c1a18e87512b82580fb23a48338f7366a95d03cd6c831`
- summary:
  `d244a26eab5fd1a3961b956058561a503269fe46b74b14dcbf9595c209c6b7ba`

## 終了判断

same-OOF上のcandidate weight、usage threshold、candidate domain、gate調整、
downstream TVT、current-test candidate生成、inference、submissionへ進めず、
fixed13 Huber hard-selector branchを閉じる。

既存の低・P4 `fixed13_selector_incumbent_reranking_instability_readout`へexp392の
独立根拠を追加する。実装・実行は別承認時だけ0 model / 0 boosterで検討する。
