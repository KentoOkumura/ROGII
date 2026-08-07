# 要件

## 依頼

train-sideで平均RMSEを改善した一方、by-well tail guardで不採用となった
exp417の固定`likpf_scale_5_x1p0`を、単体の物理モデルとしてPublic LBで記述評価する。
設計、steering、実験scaffold、バックログだけを確定し、実装・Kaggle実行・提出は行わない。

## 2026-07-31 実装承認

ユーザーの`exp452を実装してください`を、凍結済み設計に沿った実装と正規
inference Notebook採用の承認として記録する。Kaggle package作成、Kaggle実行、
submission.csv生成、competition submissionの承認には拡張しない。

## 2026-07-31 Kaggle実行承認

ユーザーの`実行してください。提出は絶対にしないでください。`を、Kaggle package作成、
canonical kernelへのpush、CPU inference実行、Kaggle outputの`submission.csv`生成・取得・
submit-checkの承認として記録する。`kaggle competitions submit`、code submission、
competition submission監視には拡張せず、提出関連flagはfalseのまま固定する。

## 2026-07-31 実行結果

private CPU inference version 1を完走し、`submission.csv`を`/tmp`へ取得して
submit-checkした。sampleとのheader、14,151行、ID順序、finite、fallback 0、
公開参照parity、SHAをすべて満たした。このCodex実行ではcompetition submissionを
開始せず、Public LBはその時点では未評価とした。

## 2026-08-01 ユーザー外部提出結果

ユーザーが外部でcompetition submissionを行い、ref `55149125`がexp452であることを
明示確認した。Kaggle statusは`COMPLETE`、Public LBは`8.797`。Codexは提出しておらず、
結果確認と記録だけを行う。同一SHA256 seed familyのarithmetic LikPF control
ref `55133074` / `9.807`より`1.010`改善したが、exp417のtail FAILを維持し、
再提出、自動昇格、LB後のパラメータ変更を行わない。

## 制約

- Route: `pf_beam`
- 1物理モデルにつき1実験、将来の実行対象となるevaluation/inference Notebookは1本だけとする。
- 候補は`likpf_scale_5_x1p0`だけとし、exp417/exp404で凍結した値を変更しない。
- PFはexp072互換の500 particles、128 seeds、`gs x1.0`を使う。
- seed集約は全suffixのseed log-likelihoodに対する
  `exp((loglik - max(loglik)) / 5)`の正規化重み付きTVT平均とする。
- seedはexp413 hidden-compatible inferenceと同じstable SHA256 per-well × seed indexを使い、
  global RNGやthread schedulingに依存させない。
- arithmetic mean、scale 3/8/12、`gs x1.3`、best seed、median、mode、medoidを生成・提出しない。
- blend、selector、gate、postprocess、ML学習、parent/control再実行を行わない。
- 公開3-well固定の行数/well数assertを禁止し、sample submission由来のID集合と
  nonempty-well contractでhidden testへ対応する。
- Public LBは候補の記述評価にだけ使い、温度、seed、particle、weight、候補式を変更しない。
- Kaggle NotebookはCPU、internet offとし、Notebook-only code competitionのhidden rerunを前提にする。
- 再現性は`docs/06_reproducibility.md`に従い、raw-test regenerated surface、
  prediction、submission、kernel versionのSHAを記録する。
- 実装、Kaggle push/run、competition submissionにはそれぞれ別のユーザー承認を必要とする。

## 受け入れ基準

- 設計段階では、候補式、seed policy、入力、禁止事項、技術gate、LB解釈規則が一意に固定されている。
- 実装後は、公開testの`likpf_scale_5`がexp413 v4参照surfaceとfloat32で最大差`0.0 ft`となる。
- `submission.csv`はsample submissionと同じID集合・行数、列`id,tvt`、重複0、欠損0、finite 100%である。
- fallback well/rowは0であり、候補以外のpredictionを代入しない。
- PF実行量、seed policy、input/schema/content SHA、prediction SHA、submission SHA、
  Kaggle kernel versionを記録する。
- gzip生成物はraw SHAに加え、decompressed content SHAを主証拠として記録する。
- Public LB確定後も自動昇格、blend作成、weight探索、追加候補生成を行わない。

## 次のアクション

Public LB `8.797`を記述censusとして記録し、exp417のtail FAILを維持したまま閉じる。
追加run、rerun、再提出、LB適応は行わない。
