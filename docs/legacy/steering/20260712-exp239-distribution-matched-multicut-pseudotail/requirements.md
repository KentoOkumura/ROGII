# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `distribution_matched_multicut_pseudotail` を
`exp239_distribution_matched_multicut_pseudotail` として実装する。旧 exp023 の
固定 3 quantile cutoff を、official-start / hidden-like な prefix 長、evaluation 長、
GR missingness、trajectory phase に合わせた deterministic multi-cut sampler へ置き換える。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 同じ source well から派生した sample を別 fold に分けない。
- outer-valid well の true tail TVT を train augmentation に使用しない。
- Public LB や current-test 3 wells に cutoff / weight を合わせない。
- 旧 exp023 LightGBM surface の再学習を目的にせず、現行 exp218 系へ移植可能な cutoff manifest と prefix replay contract を作る。
- Kaggle train push、親/control再学習、提出はユーザー承認まで行わない。

## 受け入れ基準

- raw train から official-start metadata と cutoff candidate manifest を deterministic に生成できる。
- cutoff source は prefix/eval quantile、official start までの holdout rows、GR change point、GR missing block、trajectory curvature change pointに限定される。
- wellあたりcutoff数、source別cutoff数、総 augmentation 行増幅率が config で cap される。
- target distribution と sampled distribution の prefix/eval/GR/trajectory bin差を生成物として保存する。
- official-start control、fixed exp023 3-cutoff、distribution-matched multicut の比較契約が notebook と config で追える。
- prefix依存特徴は synthetic cutoff ごとに再生成し、full-prefix cache の切り出し流用を禁止する契約を検証する。
- Jupytext percent形式のself-contained train notebookとipynbが静的検証を通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 完了判断

v11 full augmentationは15/15 boostersを完走したが、official-start OOF 8.697380066で
保存済みexp218 8.475793752から+0.221586314悪化した。受け入れ契約と再現性SHAはpassしたため、
実装不備ではなく仮説のnegative resultとして完了する。次アクションはinference/submitを行わず、
派生exp244でlate viewを扱う場合もearly-only悪化から独立した補償証拠を要求することとする。

## v12 trial submission override

2026-07-15、ユーザーからnegative CVを承知で試験提出する明示依頼を受けた。新規学習は行わず、
v11で保存済みの15 boostersをexp218の既存raw-test feature replayへ接続する。同一exp239内で
inference、submission format check、code competition submitまでを扱う。採用判断は変更せず、
LBはpseudo-tailのdistribution shift仮説を確認する補助証拠として記録する。
