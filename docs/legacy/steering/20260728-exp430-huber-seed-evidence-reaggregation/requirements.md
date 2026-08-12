# 要件

## 初回の設計依頼

exp404 と同一の x1.0 PF、500 particles、固定 128 seed を一度だけ再生し、各 seed の同一軌跡を Gaussian 尤度と Huber 尤度で再採点・再集約する実験を設計する。今回は実験ディレクトリ、steering、バックログを作り、実装・Kaggle 実行は行わない。

## 検証する問い

exp389 では HMM emission を Huber 化して平均 RMSE は改善したが tail gate を通過しなかった。一方、exp417 では Gaussian evidence による 128 seed 集約が平均を大きく改善したものの、tail は悪化した。PF の生成分布を変えず、seed 間の証拠だけを Huber 化すれば、exp417 の平均改善を保ちながら外れ seed の過信を抑えられるかを検証する。

## 制約

- Route は `pf_beam` とする。
- PF filtering、resampling、遷移、emission、粒子数、seed ラベルは exp404 x1.0 から変更しない。
- Gaussian と Huber は同一の per-seed 軌跡を使う。候補ごとに PF を再実行しない。
- Huber は exp389 と同じ `delta=1.345`、追加 clip なし、temperature は exp417 と同じ `5.0` に固定する。
- 現在保存されている exp404 生成物には per-seed 軌跡がないため、zero-PF の再集約とはせず、x1.0 trajectory bank を一度だけ再生する。
- 検証時でも後半 truth を尤度、trigger、重み生成に使わない。
- `docs/06_reproducibility.md` に従い、stable seed、入力 SHA、trajectory bank SHA、prediction SHA、実行環境を記録する。
- 実装、notebook 本体、Kaggle push、学習・推論は本設計の対象外とする。

## 受け入れ基準

- 親実験、固定式、固定 128 seed、PF 実行量、比較対象、技術 gate、科学 gate が文書と `config.yaml` で一致している。
- Gaussian control と Huber candidate は同一 trajectory bank から生成される。
- full run は 1 PF variant、773 well、98,944 seed-well trajectories、49,472,000 particle starts と明記されている。
- 親 control の独立再学習・独立再実行を含まない。
- 実装前であること、実行結果がないことを README、SESSION_NOTES、result、metrics に明記している。

## 非目標

- exp389 の HMM 結果を成功扱いに変更しない。
- Huber delta、temperature、PF scale、粒子数、seed 数を探索しない。
- affine、AR(1)、self-GR、datum reinjection と組み合わせない。
- 本設計だけを根拠に deterministic anchor や提出候補へ昇格しない。

## 2026-07-28 実装依頼

ユーザーの「exp430を実装してください」を受け、上記の固定設計を変更せず、
Jupytext起点のcompact self-contained train / inference、fixed4 preflight、
full4 shard、truth-late merge、専用contract testまでを実装対象とする。
Kaggle package、push、run、inference、submissionは実装依頼の承認範囲外とする。
