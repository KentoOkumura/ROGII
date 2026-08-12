# 要件

## 依頼

`adaptive_robust_likelihood_pf` を temperature 専用の PF/Beam 本実験として実装する。GR observation が局所的に信用しにくい row だけ particle likelihood を緩め、誤った GR motif による particle collapse を抑える。

## 制約

- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`。保存済み `likpf_mean` を `T=1` control とし、再生成しない。
- 新規 variant: `T=2` と `T=4` のみ。outlier mixture、transition、particle数、seed数、resampling、Beam は変更しない。
- gate は current/past GR、typewell GR、observed prefix、pre-update particle stateだけを使う。true TVT、target、error、oracle、LB を使わない。
- 再現性は `docs/06_reproducibility.md` に従い、per-well / variant / seed index の stable SHA256 seed、単一 worker、入力・出力 SHA を記録する。
- CPU-only Kaggle train-side audit とし、inference、submission、GPU、LightGBM は範囲外とする。

## 受け入れ基準

- 773 well の exp072-compatible pseudo-tail で `T=2/4` を評価し、saved `exp072_likpf_mean` と比較できる。
- overall、distance bucket、hidden-like、by-well、ESS、resampling、gate rate、sampled particle p05-p95 coverage、first sampled loss を保存する。
- gate 外で `T=1` が使われること、outlier mixture が含まれないことを設定・診断で確認できる。
- direct replacement、inference、submission は coverage と RMSE/worst-well guard が成立するまで作成しない。
