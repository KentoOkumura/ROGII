# 要件

## 依頼

`exp413_scale5_likpf_full_replacement_on_exp335` を TVT 予測のベースとし、
Stage C の既存 selector compact 74 列を
`exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264` の fixed13 selector
compact 77 列へ差し替える実験を設計する。selector を追加するのではなく、対応する
既存ブロックを置換する。

初回ターンの範囲は backlog、steering、実験ディレクトリ、設計確定までとし、実装、
Kaggle package、学習、推論、submission は行わない。

## 2026-08-02 実装承認追記

ユーザーの「exp502を実装してください」によりtrain-side実装と契約テストを承認した。
別名compact self-contained Jupytext sourceと候補notebookの作成までを対象とし、正規
notebook採用、Kaggle package、学習、推論、submissionは引き続き未承認とする。

## 仮説

exp501 fixed13 selectorのfold-safe compact77は、selector-level tail FAILを維持したままでも、
exp413 TVTモデルにとって既存nested74より有用な表現になり、保存exp413 OOFを改善し得る。

## 制約

- Route: `ml_model`
- 親実験 / control は `exp413` の保存済み Stage D OOF とする。
- treatment の特徴量は `clean273 + exp501 compact77 + exp413 signed23 = final373` とする。
- `exp413 nested74` と `exp501 compact77` を同時に残す add-only、concat、blend は禁止する。
- `clean273`、`signed23`、outer 5 folds、TVT LightGBM 3 configs、seed、target、
  score rows、early stopping、後処理、評価 scope は `exp413` から変更しない。
- exp501 selector 40 本、exp413 signed selector 20 本、exp490 HMM、PF、Beam、
  exp413 control は再学習・再生成しない。
- 将来の train は treatment 1 variant × LightGBM 3 configs × outer 5 folds =
  15 GPU boosters のみとする。これはコスト契約であり実行承認ではない。
- exp501 の fixed13 hard OOF 改善と Stage C tail gate FAIL の両方を入力証拠として保持し、
  この実験の作成によって exp501 の terminal decision を再分類しない。
- current-test feature generation、inference、submission は train-side gate 判定後の別承認とする。
- 再現性は `docs/06_reproducibility.md` に従い、入力 SHA、fold manifest、
  feature schema/content、model manifest、OOF prediction、Kaggle kernel version を記録する。

## 受け入れ基準

- 置換面が `exp413 nested74 -> exp501 compact77` の 1 差分として明記されている。
- old selector 74 列が final matrix に 0 列、new selector 77 列が 77 列であることを
  technical gate に含める。
- final schema が順序固定の 373 列で、重複名、欠損 join、row/fold mismatch が 0 である。
- exp501 と exp413 の nested fold manifest SHA
  `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`
  の一致を入力 gate にする。
- control は exp413 saved OOF RMSE `7.884802794404715`、OOF SHA
  `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`
  とし、再学習 0 本で比較する。
- promotion gate、tail readout、PASS/FAIL 後の動作が事前固定されている。
- 実験ディレクトリの canonical train / inference notebook は placeholder のままで、
  別名train候補だけが実装され、Kaggle packageは作られていない。
- deterministic anchor として扱う場合は feature content SHA、model SHA、prediction SHA、
  submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed content SHA を
  主証拠として記録している。

## 次のアクション

train-side実装と静的検証で停止する。正規notebook採用、Kaggle package/train、
inference、submissionはそれぞれ別途承認後に進める。
