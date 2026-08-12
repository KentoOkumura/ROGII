# 要件

## 依頼

`multimode_pfbeam_local_correlation_audit` を実装する。PF/Beam が局所相関ずれ候補を複数 mode として保持できているか、または resampling / pruning で早期に単一 mode へ潰れているかを train pseudo-tail で診断する。

## 制約

- Route: `pf_beam`
- 親実験: `exp142_trajectory_aware_pf_transition_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 提出候補は作らない。train-side diagnostic only。
- true TVT は評価・oracle 診断にだけ使う。local GR correlation と mode 診断は target-free 入力から計算する。
- 再現性: `docs/06_reproducibility.md` に従い、PF stochastic seed、並列実行、gzip decompressed SHA、Kaggle bootstrap の扱いを記録する。

## 受け入れ基準

- exp143 の `config.yaml`、train/inference notebook、補助 `.py`、README、SESSION_NOTES、result、metrics 初期値が揃っている。
- strict exp072 PF-Z parity、strict multiseed、multimode PF variants、既存 exp072 candidates を同一 pseudo-tail rows で比較できる。
- `candidate_metrics.csv`、`bucket_metrics.csv`、`by_well.csv` に候補 TVT の基本評価を保存する。
- `strict_pf_z_quality.csv` と `multimode_pf_z_quality.csv` に ESS、resampling、collapse、mode count、mode entropy、seed spread、local GR-correlation topK spread を保存する。
- `candidate_wide.csv.gz` に row-level mode/correlation 診断列を保存する。
- gzip 生成物は raw `.csv.gz` SHA だけでなく decompressed content SHA を summary JSON に記録する。
- inference notebook は train-side diagnostic only であることを明示し、submission を作らない。
