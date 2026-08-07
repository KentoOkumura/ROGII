# 要件

## 依頼

- `exp226` の予測値を使わず、同実験の「井戸を区間分割し、空間的に近い区間の傾きを補間する」という考えだけを exact HMM に組み込む。
- 事前 readout / Stage 0 を挟まず、最初の科学実行から HMM を回す設計にする。
- 今回は backlog、steering、実験ディレクトリ、比較条件、成功条件までを確定し、実装、Kaggle push、実行、推論、提出は行わない。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` の exact HMM とし、GR/typewell emission、状態格子、`sig_r`、`sig_p`、momentum、初期分布、posterior mean 出力を固定する。
- `exp226` の OOF、`tvt_geop`、`tvt_pred`、GR 補正、adaptive kappa、near-strike ANCC、U projection、保存済み予測を読み込まない。
- donor は outer-train well の raw train truth のみとし、outer-valid fold の全 well を donor 集合から除外する。
- target の未知 suffix の TVT、誤差、oracle 情報、formation、well-id 固有ルールを遷移平均の生成に使わない。
- 変更は「K=16 の donor local slope field から作る時変 rate-prior mean」の 1 変数に限定し、parameter grid、blend、transition variance 変更を行わない。
- control HMM は再実行せず、SHA 固定済み exp209 cache と既存 score を比較対象にする。
- 実装時の計算契約は scientific variant 1、reporting folds 5、HMM well-runs 773、LightGBM config 0、trained fold 0、booster 0、GPU 0 とする。

## 受け入れ基準

- 設計文書、`config.yaml`、実験 README、`SESSION_NOTES.md`、`result.md`、`metrics.json`、`KAGGLE_DIRECTION.md`、`experiment_summary.md` が同じ契約を示す。
- primary は candidate HMM 単体とし、exp209 exact HMM の direct RMSE `11.938287417 ft` に対し pooled `0.05 ft` 以上改善し、5 fold 中 4 fold 以上で改善する。
- 1000+、hidden-like spatial、hidden-like typewell-purged、by-well p95 を非悪化、worst-well 回帰を `+0.25 ft` 以下に保つ。
- 技術 gate と科学 gate は AND とし、不通過時に K、近傍数、bandwidth、ridge、fallback、HMM parameter を同一 OOF 上で救済しない。
- 追加実装指示前の状態は `design_frozen_not_implemented` とし、当時のplaceholder notebookを
  実装済みと扱わない。
- 将来 deterministic anchor として扱う場合は、raw input identity、donor-segment / rate-schedule content SHA、prediction SHA、Kaggle kernel version を記録する。学習 model と submission を生成しない段階では model SHA / submission SHA は非該当と明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 2026-07-23 追加実装指示

- ユーザーの `exp362を実装してください` を受け、設計凍結済みの科学契約を変更せず
  compact self-contained train 候補を実装する。
- 既存正規 `.ipynb` は明示的な上書き承認がないため保持し、Jupytext percent source と別名
  compact `.ipynb` を作成する。
- inference は無効のまま、sample submission copyを生成しない fail-closed compact 候補にする。
- 実装完了条件は、stable fold、fold-safe donor、K16 local gradient、residual-rate exact HMM、
  prediction freeze、late truth/control join、全gate/SHA出力、専用test、Jupytext、構文、
  Ruff F821、strict experiment validationが揃うこととする。
- Kaggle package/push/run、正規notebook採用、inference、submissionは追加実装指示に含めない。
