# exp072_exp063_full_replay_feature_cache

## 状態

- ルート: `pf_beam`
- 状態: `implemented`
- 親: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 作成日: 2026-06-14

## 目的

exp063 と同じ full public replay train features を CPU-only Kaggle Notebook で再生成し、後続実験が train feature cache として再利用できるようにする。

この実験はモデル学習を行わない。test features は hidden/current test に依存するため、この実験では作らず、各後続実験の inference notebook 内で再生成する。

## 仮説

exp063 full replay train features を CPU-only notebook の出力として固定すれば、後続の LightGBM 実験で重い PF/Beam/likelihood-PF train feature generation を繰り返さず、GPU quota をモデル学習と推論に集中できる。

## 検証方針

Kaggle CPU notebook で raw train files から `pixiux_likpf_public_replay` の full train feature frame を生成する。生成後の summary で rows、well count、feature count 196、feature SHA を確認する。test features は保存せず、各 downstream inference notebook で current raw test files から再生成する。

## 生成物

- `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- `exp063_full_replay_feature_cache_feature_schema.csv`
- `exp063_full_replay_feature_cache_summary.json`

期待 feature count は exp063 `pixiux_likpf_public_replay` と同じ 196。

## 所見

実装済み。Kaggle 実行はまだ行っていない。
