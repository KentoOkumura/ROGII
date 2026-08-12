# 設計

## 仮説

exp410で見えたresampling roughening 10倍のparticle basin維持効果が全773 train
wellsへ一般化し、exp072保存controlに対して事前固定したfold / scope / well-tail /
persistent-offset guardをすべて満たす。

## 実装承認追補

初回設計時の承認範囲はdesign-onlyだった。2026-07-27のユーザー依頼
`exp416を実装してください`により、compact self-contained train候補とcontract testsの
実装まで追加承認された。正規Notebook採用、Kaggle package、push、run、inference、
submissionは引き続き未承認である。

## 実行承認追補

2026-07-27のユーザー依頼`実行してください`により、正規train Notebook採用、
Kaggle package作成、CPU 4 shardのpush / run、strict mergeとtrain-side評価まで
追加承認された。固定probe rerun、raw-test inference、submissionは承認範囲に含めない。

## 結論

exp410の12 target-late wellsではroughening 10倍がepisode SSEを
`0.752997倍`へ下げ、10/16 episodes、8/12 wellsを改善した。一方、well符号検定は
`p=0.3877`で、within-seed / explicit resampling原因では悪化した。したがって
prediction候補として一般化したとは扱わず、exp072 controlを再実行しない全OOF単一介入で
判定する。

## アプローチ

1. exp072と同じraw train、Type Well、stable 128-seed契約を読む。
2. 各wellでexp072 kernelを1回だけ再生し、resampling後のroughening振幅だけを
   position `1.00 ft`、rate `0.010`へ固定する。
3. candidateの全row identity、予測、well manifest、runtime、code/config/input SHAを
   truthを読む前にfreezeする。
4. exp072 cacheの`last_known_tvt + likpf_mean_d`をcontrolへ復元し、exp209の
   exact reconstructed `likpf_mean`とID一対一・row-level parityを確認する。
   control PFは再生しない。
5. freeze後にsuffix truth、reporting fold、hidden-like role、exp410 episode identityを
   結合し、事前固定したpooled / fold / scope / well-tail / persistent-offset guardを判定する。
6. 4 shardはsuffix row数のdeterministic LPTで分ける。各well seedはshard番号や実行順を
   含めず、shard間で独立に再現できるようにする。

## 実験範囲

- 対象実験: `exp416_roughening_x10_likpf_full_oof_ablation`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 原因監査: `exp410_likpf_particle_resampling_basin_audit`
- 変更する変数:
  - resampling position roughening: `0.10 -> 1.00 ft`
  - resampling rate roughening: `0.001 -> 0.010`
- 固定する変数:
  - particles 500 / seeds 128 / seed index 0--127
  - initial spread 4.5 ft / initial rate spread 0.01
  - momentum 0.998 / rate noise 0.002 / position noise 0.005
  - ESS threshold 0.5 / systematic resampling
  - Gaussian raw-GR emission / sigma clip `[10,60]`
  - Type Well grid 0.2 ft / pad ±100 ft / GR missing補間
  - arithmetic seed mean
- control: 保存済みexp072 `likpf_mean`
- candidate: `likpf_roughening_x10_mean`

## 実行量

- scientific PF variants: 1
- candidate PF well-runs: 773
- control PF well-runs: 0
- seed-well trajectories: `773 ×128 = 98,944`
- particle starts: `98,944 ×500 = 49,472,000`
- reporting folds: 5（学習foldではない）
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`
- Kaggle CPU shards: 4
- 保守的runtime: 各shard4時間、hard stop各9時間

## 評価と固定gate

control RMSEは`11.594897672217703 ft`を期待値とし、primary gainは
`RMSE(control) - RMSE(candidate)`とする。

全条件をANDで要求する。

1. pooled gain `>=0.05 ft`
2. 改善fold `>=4/5`
3. raw-GR observed gain `>=0.05 ft`
4. raw-GR missing、1000+、hidden-like spatial / typewell-purgedのregression `<=0`
5. by-well delta RMSE p95 `<=0`
6. worst-well regression `<=0.25 ft`
7. exp410 persistent-offset episode SSE reduction `>=5%`

FAIL時は倍率、position/rate別、process noise、ESS threshold、GR sigma、seed /
particle数、well/row gateを同じOOFで探索しない。

## 実装時のNotebook契約

初回設計時点では実装しない契約だった。追加の実装承認後、Jupytext percent形式の
compact self-contained train候補を作成した。正規Notebookは採用承認までplaceholderの
ままにする。

Notebook上で少なくとも次を追える構成にする。

1. Imports
2. Runtime / stable-seed / SHA helpers
3. Raw and saved-control input checks
4. Exact exp072 PF kernel
5. Roughening-x10 one-factor contract
6. Shard selection and candidate generation
7. Prediction freeze and late truth join
8. Fold/scope/tail metrics, manifests, and gate

## 再現性設計

- seed policy:
  `sha256("likpf::train::<well_id>") % 2147483647 + 1 + seed_index`
- stochastic処理:
  initialization、transition noise、systematic resampling、roughening。
- RNG順:
  exp072と同じcall順を維持し、振幅だけを変える。diagnostic用の乱数を追加しない。
- 並列処理:
  well内Numba single worker。well seedはimmutable key由来なのでshard順に依存しない。
- CPU/GPU:
  Kaggle CPU、GPU off、internet off。
- SHA:
  raw input、decompressed input、schema、code、config、well manifest、
  prediction raw/decompressed/logicalを記録する。
- rerun:
  固定probe wellを別runで再実行し、candidate predictionのcontent parityを確認する。
- deterministic anchor:
  full coverage、全SHA、probe rerunが揃うまでは主張しない。

## リスク

- 選択バイアス:
  exp410 sentinelはtarget-lateなので、全773 wellsだけをprimaryにする。
- 因果混同:
  rougheningが将来のRNG trajectoryを変える効果を含む。即時resampling extinctionだけの
  介入とは解釈しない。
- tail:
  exp410では一部causeを悪化させたため、pooled改善だけで昇格しない。
- runtime:
  500×128×773は高コスト。controlを再実行せず、candidateを4 shardへ固定分割する。
- 再現性:
  float dtype、seed `+1`、Numba RNG、input sourceの違いでparityが崩れる可能性がある。
  preflightとstrict mergeでfail closedする。
- CV/LB:
  train-side OOFの単一PF候補であり、PASSしてもinference / submissionは別承認とする。

## 対象外

- process noise 3倍との同時比較
- roughening倍率grid
- adaptive roughening / well gate / row gate
- seed aggregation変更
- HMM / Beam / ML / blend
- raw-test inference / submission
