# 設計

## 構造

exp238の保存済みfinal `lgb_mean` OOFをbaseとする。outer foldごとに、inner 4 selector平均で作られたrole=`valid`候補誤差scoreからtop1、top2、likPFとの差を計算する。row gateとtarget-free well gateを通過したrowだけ、次式でtop1候補方向へ補正する。

`prediction = base + alpha * clip(top1_candidate - base, -candidate_delta_clip, candidate_delta_clip)`

alphaとclipは3つの固定profileに置き、最大移動量を4/7.5/12 ftへ制限する。well gateは候補familyのdominant shareと補正方向consistencyだけを使い、truthを見ない。

## leakage防止

- outer-valid rowの候補選択は、そのwellを学習に含まないouter別selector scoreだけを使う。
- candidate pathsは既存group-safe OOF生成物、baseはexp238 fold-held-out OOFを使う。
- truthは全variant予測を固定した後のmetric計算だけで読む。
- profileはKaggle実行前に固定し、結果に応じて同notebook内で閾値探索しない。
- hidden-like assignmentはsubgroup metricだけに使う。

## 再現性

- 新規RNG、PF/Beam生成、model fitなし。
- exp238 selector scores、final OOF、candidate source、config、outputのSHAをsummaryへ保存する。
- gzipはdecompressed SHAを記録する。
- Kaggle package bootstrapのconfig/source SHAとpush後metadataを確認する。

## 実行コスト

- active audit 1、fixed profile 3、model config 0、fold training 0、booster 0。
- CPU、single process、parent/control再学習なし、submissionなし。
- 5個のfull-row score gzipを逐次chunk読込し、score matrixだけをfloat32で保持する。

