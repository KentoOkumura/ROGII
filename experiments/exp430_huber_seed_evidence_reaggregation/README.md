# exp430_huber_seed_evidence_reaggregation

## 状態

- ルート: `pf_beam`
- 状態: merge technical PASS / scientific FAIL、terminal close
- CV: Huber `12.992940` / matched Gaussian `12.999103` / 保存exp404 `10.914522`
- Public LB / Private LB / Submit ID: なし
- 作成日: 2026-07-28
- 親実験: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- 比較根拠: `exp417_scale5_seed_aggregation_promotion_audit`

## 仮説

exp404 x1.0 の固定 128 seed PF 軌跡を一度だけ生成し、同じ軌跡を
Gaussian と Huber `delta=1.345` で再採点すれば、exp417 の平均改善を保ちつつ
seed weight の外れ値と per-well tail 悪化を抑えられる。

## 実装

- exp404 と同じ x1.0 GR scale、500 particles、128 seeds、Gaussian particle
  filtering、transition、resampling、roughening、RNG 消費順を維持する。
- per-seed 軌跡は `float64` の `.npy` bank に保存し、logical SHA を確定してから
  evidence を計算する。
- `gaussian_matched` と `huber_delta_1p345` は同じ bank SHA を参照し、
  centered softmax / temperature `5.0` で集約する。
- arithmetic mean と exp404 parent marginal-Gaussian replay は技術比較だけに使う。
- 実行は `preflight → full_shard 0..3 → truth-late merge` の三段階。
- full shard は SHA 固定済み preflight PASS を必須とし、merge は4 shardの
  summary SHAをすべて固定しない限り開始しない。
- inference notebook は train-side gate PASS と別承認まで必ず停止する。

## 実行量

- technical preflight: 1 PF variant、4 wells、4 PF well-runs、
  512 seed-well trajectories、256,000 particle starts
- full: 1 PF variant、773 PF well-runs、98,944 seed-well trajectories、
  49,472,000 particle starts、4 CPU shards
- 親 full control 再実行: 0
- LightGBM config / fold training / booster / model / HMM / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 検証方針

- 親 exp404 PF kernelとの小配列bit parity
- fixed 4 wellでparent marginal replayと保存exp404 T=5、arithmetic meanと
  保存exp404 arithmetic列を照合する。保存exp072 delta再構成との差は丸め診断だけにする
- 同じ frozen bank を1 worker / 4 workersで再採点した prediction / evidence SHA parity
- truth、fold、hidden-like roleは4 shardのprediction/evidence SHA freeze後にだけ読む
- Huberをmatched Gaussianと保存exp404 T=5の双方に対してoverall、fold、
  deep/shallow、missingness、roughness、hidden-like、by-well tailのAND gateで判定

roughness は、frozen arithmetic-mean trajectory のwell別二階差分RMSを全well中央値で
二分するtarget-free scopeとして実装した。

## 所見

同じfloat64 trajectory bankを先に凍結するため、Huberの効果をPF生成分布の変更と
混同せずに判定した。Huberはmatched Gaussianを`0.006164 ft`だけ改善したが、
固定`0.10 ft` gateに届かず、fixed scopeとby-well tailもFAILした。さらに
保存exp404 temperature-5より`2.078417 ft`、arithmetic meanより`1.398042 ft`
悪く、trajectory-residual evidence familyはparent marginal evidenceを代替しない。

## 実行入口

- train notebook:
  `exp430_huber_seed_evidence_reaggregation_train.ipynb`
- inference notebook:
  `exp430_huber_seed_evidence_reaggregation_inference.ipynb`
- compact Jupytext source:
  `exp430_huber_seed_evidence_reaggregation_compact_selfcontained_train.py`
- Kaggle: merge version 1、id_no `129051025`

## 次

version 1のbinary float32 / CSV再読込float64 comparator不整合を、
toleranceを緩和せず親保存dtypeへ正規化して修正した。version 2は12 / 12
technical checks PASSで、v1/v2のtrajectory、prediction、evidence raw SHAも一致した。
preflightとfull 4 shardを完了し、truth-late mergeのtechnical gateは全PASSした。
科学gateはFAILのため`huber_seed_evidence_reaggregation_rejected_close_without_rescue`
で閉鎖する。delta / temperature / clip / scale / particle / seed / filtering尤度、
well gateのsame-OOF救済、inference、submissionは行わない。
