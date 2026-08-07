# exp263_last_anchor_better_candidate_confidence_pair_cache

## 状態

- ルート: `pf_beam`
- 状態: Stage 0完了・Stage 1値parity／namespaced confidence parity完了
- CV: 新規学習なし。reference anchorはexp072 `last_anchor` 15.909866082
- Public LB: 7.800（v2固定blend）
- Private LB: なし
- Submit ID: `54761954`
- 作成日: 2026-07-16
- 親実験: `exp072_exp063_full_replay_feature_cache`
- Kaggle kernel: `kentookumura/exp263-last-anchor-pair-cache-train` version 1

## 仮説

`last_anchor`より良いknown pathをfamily圧縮し、候補値だけでなく保存済みsourceに実在する
target-free confidence、候補間disagreement、固定formulaを一度だけID整合・SHA固定すれば、後続
selector / ML / fixed blendが同じprimitive入力を安全に再利用できる。

## 変更点

- known 33候補をreference catalogとして固定し、row cacheはcore 12候補だけに限定した。
- coreのraw-test-ready tier 6本とtrain-only / diagnostic tier 6本を分離した。
- confidenceは実source列だけを保存し、未提供診断はNaN + valid/missing contractにした。
- outer-fold eligibilityをouter-valid foldを除いた4 foldsだけで計算する。
- 有望8 pair、`blend_likpf_hmm_w500` alias、3 named combinationを固定した。
- pair/triple tensorは保存せず、`CandidateCache`がfold/chunk単位で再構成する。
- Stage 1は6 raw-test-ready primitive、5 pair、固定`exp226_w500_50_50`だけを扱う。
- Stage 1は6 primitiveのsource-native confidence 21列を候補別namespaceで同時出力する。

## 検証方針

- Fold: well groupを行数balancingしたdeterministic 5 folds
- Group: `well`
- Metric: RMSE
- Leakage check: row cacheへtarget/error/oracle/selector outputを入れず、target-derived値はcatalog、
  pair readout、eligibility manifestに隔離する。
- Reproducibility: source file SHA、gzip decompressed content SHA、logical Parquet content SHA、schema
  SHA、formula DAGを記録する。

## 実行入口

- Stage 0 notebook: `exp263_last_anchor_better_candidate_confidence_pair_cache_train.ipynb`
- Stage 1 notebook: `exp263_last_anchor_better_candidate_confidence_pair_cache_inference.ipynb`
- Loader: `candidate_cache_loader.py`
- Loader仕様: `loader_contract.md`
- Schema: `cache_schema.json`
- Kaggle準備:
  `make prepare-kaggle-notebooks EXP=exp263_last_anchor_better_candidate_confidence_pair_cache EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp263-last-anchor-pair-cache-train --title 'exp263 last anchor pair cache train' --run-on-push --strict --no-src"`

## 実装時の学習コスト

| 項目 | 数 |
| --- | ---: |
| active variant | 0 |
| LightGBM config | 0 |
| fold training | 0 |
| booster | 0 |
| parent/control再学習 | 0 |

## 現在の結果

Kaggle CPU version 1は3,783,989行 / 773 wellsを951.444秒で処理し、candidate value 60、
confidence 60の計120 Parquet partition（約3.0 GiB）を生成した。cache manifest SHAは
`85e60ac1...a26bb9e`、catalog SHAは`7cd74866...e9e6e0`。source 9群 / 12 gzipは全行coverageし、
raw SHAとdecompressed content SHAをmanifestへ固定した。Stage 1 inference v2は14,151行 / 3 wells、
6 primitive / 5 pair / fixed formulaを225.459秒で完走し、固定blendを提出してPublic LB 7.800だった。
v3は21 namespaced confidence列を含む14,151行 × 36列を354.341秒で完走し、旧15値列はv2と
exact一致、拡張Parquet SHAは`bda05028...e360c`。submissionはv2とbyte-identicalのため再提出していない。

## 所見

実装契約の33/12/6/8/3境界、target-derived readoutの隔離、deployability tier、formula DAGを
full cacheでも確認した。代表value/confidence 4 Parquetはrow/bytes/file/content/schema SHAがmanifestと
一致し、best pairのvirtual loader再構成も757,738行で最大誤差0だった。

## リスク / 注意

- HMM+LGB `exp221/234/240`とselector / TVT model outputsはscope外。
- exp104 PF-Z seedbag 5本はsuperseded referenceであり、row cacheもpair sweepも作らない。
- K8 m1-m6はexternal diagnostic bankに残し、core cacheへ複製しない。
- `w500`をselectableにする場合、親`likpf_mean` / `exact_hmm`を同時selectableにしない。
- Stage 0は後続OOF cacheのcanonical inputとして利用できる。raw-test selector用の21列coverage/SHAも
  Stage 1 v3で確認済み。ただしselector学習はexp264の別承認とする。

## 次

1. exp263はStage 1まで完了とする。
2. exp264 Stage Bの10 CPU boostersは別承認を得て実行する。
