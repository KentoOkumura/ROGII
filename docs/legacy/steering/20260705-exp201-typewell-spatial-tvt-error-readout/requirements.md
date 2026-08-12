# 要件

## 依頼

exp148 OOF の誤差について、次の傾向を診断する。

- 既存実験で見つかっている 50 数個の共通 typewell group ごとに、誤差の符号、大きさ、残差形状が似ているか。
- XY 座標が近い well 同士で、bias、RMSE、残差形状が似ているか。
- true TVT が急激に上昇または下降する well / 区間で、予測がなめる、遅れる、オフセットするなどの傾向があるか。
- TVT 予測が well 全体で上振れ/下振れする offset well が、typewell group や XY 近傍に偏るか。

## 制約

- Route: `ml_model`
- 新規学習、推論、提出は行わない。既存 OOF prediction と raw train truth を読む診断実験とする。
- 主入力は exp148 train v1 の `lgb_mean` OOF prediction と、`native_overlap_1` の 54 共通 typewell group 対応。
- OOF baseline / RMSE は診断値であり、route anchor 更新根拠として扱わない。
- 再現性: deterministic な集計のみ。gzip 入力は decompressed content SHA を記録する。

## 受け入れ基準

- well 別、typewell group 別、XY 近傍、急変 TVT、offset well の CSV が生成されている。
- 残差形状の類似度について、同一 typewell 内、XY 近傍、全体平均との差が記録されている。
- 急変 TVT 区間で true step と predicted step の damping / error が集計されている。
- `result.md` に、強い傾向、弱い傾向、次に実験化するなら何を feature / gate にするかを日本語で記録する。
- gzip 入力は raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
