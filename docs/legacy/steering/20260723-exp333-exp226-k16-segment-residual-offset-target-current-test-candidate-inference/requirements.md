# 要件

## 依頼

exp361 で確認した exp333 の add-one candidate novelty を受け、同じ
`exp333_exp226_k16_segment_residual_offset_target` 内で保存済み Stage 1
5 fold model を current test に適用し、候補パス artifact を生成する。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 新規学習、control / parent 再学習、booster 追加は行わない。
- variant は `exp333_k16_segment_residual_offset` の 1 個、保存済み model は
  outer fold 0..4 の 5 個、推論時はその等重み平均に固定する。
- exp226 current-test v1 の 14,151 行を base とし、同一 K16 segment 定義、
  exp072 deterministic raw-test replay、U projection、GRWR の契約を維持する。
- offset の clip / shrink / taper / interpolation / slope は追加しない。
- selector、fixed12 average、blend weight 探索、submission file 生成、
  competition submit は行わない。
- 元の direct-promotion `FAIL_CLOSE_BRANCH` は履歴として維持し、今回の許可は
  exp361 が支持した add-on candidate artifact 生成だけに限定する。

## 受け入れ基準

- Kaggle CPU inference が COMPLETE し、sample submission と同一順序・一意 ID の
  14,151 行、3 well を候補 artifact に保存する。
- exp333 train model manifest SHA、5 model SHA、136 feature の名前と順序、
  saved train summary の固定値を同一 Notebook 内で照合する。
- current-test row feature schema/content、segment feature content、fold component
  prediction、ensemble prediction、candidate file の SHA を保存する。
- base / offset / candidate が全行 finite で、各 well に segment 0..15 が存在し、
  segment境界・row coverage・fold ensemble の technical guard を通す。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、
  prediction SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
- `submission.csv` と submission SHA は生成しない。
