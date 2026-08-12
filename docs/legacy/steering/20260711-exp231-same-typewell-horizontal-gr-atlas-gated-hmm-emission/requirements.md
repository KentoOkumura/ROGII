# 要件

## 依頼

`same_typewell_horizontal_gr_atlas_gated_hmm_emission` を `exp231` として実装する。

同じ exp065 native-overlap typewell group に属する他 horizontal well の train-fold GR と TVT から、`(typewell group, TVT bin) -> local GR patch distribution` atlas を作る。exp209 exact HMM の transition・state grid・base typewell-GR emission は固定し、atlas は target-free confidence が高い row にだけ小さい補助 emission として加える。

## 制約

- Route: `pf_beam`。
- HMM の primary parent は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- CV は fixed-seed well GroupKFold 5 folds。validation well および同 fold の全 validation well を atlas の source pool から完全に除外する。
- source peer の true TVT は training-fold atlas の bin index にだけ使う。validation true TVT は target と結果診断だけに使う。
- GR patch window は `64/128/256` rows を robust normalize し、16点に再標本化する。
- `final_loglik = base_loglik + alpha * confidence * centered_peer_score` とし、`alpha=0.01/0.025/0.05` のみ比較する。
- confidence は self match/novelty、peer support、TVT uniqueness、base emission ambiguity、innovation、GR change point だけで作る。validation label、true error、oracle gate は禁止する。
- peer score は state 方向に中心化・clip する。全 state に一律の尤度シフト、TVT direct replacement、PF weight 変更、blend、selector 化は禁止する。
- saved exp072 cache を比較に使い、exp072 / exp209 control を再生成しない。LightGBM config は 0、booster は 0。
- 最初の scope は Kaggle train-side cache/OOF readout のみ。raw-test atlas、inference、submission は作らない。

## 受け入れ基準

- exp231 の steering、config、train/inference notebook、helper、README、SESSION_NOTES、result、metrics が整合する。
- fold ごとの atlas source / validation exclusion / cluster assignment SHA を summary に残す。
- global RMSE、distance bucket、hidden-like、worst-well、HMM uncertainty、step delta、candidate true-state rank を出力する。
- peer confidence による persistent-offset onset AUC と q90 lift を、label を generation に使わず診断として出力する。
- `validate-exp`、Python 構文チェック、Jupytext conversion/test、F821 lint が通る。
- gzip feature cache は decompressed content SHA を主証拠として記録する。
