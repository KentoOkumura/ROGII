# 要件

## 依頼

`spatial_neighbor_prior_signal_audit` を新規実験として実装する。X/Y が近いだけでなく、掘削方向、軌跡形状、local tangent、dZ/dMD、prefix TVT range が近い train wells から fold-safe spatial neighbor prior を作り、PF/Beam/likPF の TVT 誤差方向を説明できるか診断する。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親 cache は `exp099_pf_multi_observation_likelihood_probe` の train-side feature cache を使う。
- validation well 自身、および同 fold valid wells の true TVT を neighbor source に入れない。
- `xy_only` は control として扱い、主仮説は direction / shape 類似を含む variant とする。
- train-only formation columns は使わない。
- direct correction submit はこの実験では選ばない。

## 受け入れ基準

- `experiments/exp114_spatial_neighbor_prior_signal_audit/` に `config.yaml`、train notebook、補助 audit script、初期記録がある。
- train notebook は設定、入力 cache、variant、audit 実行、生成物確認をセルで追える。
- audit script は `candidate_metrics`、`signal_metrics`、`bucket_metrics`、`by_well`、`neighbor_summary`、`oof_predictions`、`summary.json` を保存する。
- `likpf_mean` / `pf_ancc` / `beam_mean` に対して prior-only、clipped correction、base error と prior-base 差分の相関、符号一致率を評価できる。
- deterministic anchor としては扱わず、Kaggle train 実行後に入力 SHA、decompressed OOF SHA、prediction SHA を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
