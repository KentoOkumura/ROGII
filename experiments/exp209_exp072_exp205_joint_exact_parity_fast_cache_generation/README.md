# exp209_exp072_exp205_joint_exact_parity_fast_cache_generation

## 目的

exp072 full replay feature cache と exp205 exact HMM cache / direct comparison を同一 Kaggle train notebook で生成し、出力 parity を維持したまま wall time を短縮する。

最初の高速化は、exp072 の full cache を従来どおり書き出したうえで、comparison 必要列を exp205 direct comparison に in-memory で渡し、exp205 側の 2GB gzip 再読込を避けることに限定する。likPF-only/slim cache 化はこの実験では扱わない。

## 状態

- Kaggle train v3 完了、metric parity は近似一致として許容
- CPU-only train feature generation audit
- 推論、提出、raw-test regeneration は対象外

## 仮説

exp072 full cache と exp205 HMM cache の生成値を変えずに同一 notebook で連結し、exp072 generated DataFrame の comparison 必要列を direct comparison へ渡せば、exp205 側の 2GB gzip 再読込を省ける。HMM outer parallel は serial parity 確認後にのみ試す余地がある。

## 範囲

- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- 親: `exp205_exact_hmm_smoother_exp072_compatible_cache_audit`
- 推論、提出、raw-test regeneration は対象外。

## 主な生成物

- `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_amerhu_exact_hmm_smoother_default_train_features.csv.gz`
- `exp209_vs_exp072_exp205_summary.json`
- `exp209_joint_generation_summary.json`

## 検証方針

- exp072 generated cache の rows / wells / schema / SHA を exp072 v2 reference と比較する。
- exp205 HMM generated cache の rows / wells / schema / decompressed SHA を exp205 v2 reference と比較する。
- direct comparison の best candidate / RMSE が exp205 v2 と一致するか確認する。
- wall time が `< 6h` に入るか確認する。serial parity が通った場合だけ `outer_workers=2` を benchmark する。

## 実行方針

既定設定は parity 優先で HMM outer loop を serial にしている。serial fast path が exp072/exp205 v2 parity を満たした場合だけ、`feature_cache.hmm.outer_workers=2` を試す。

## 所見

v3 は `kentookumura/exp209-joint-exact-parity-train` で完了。HMM cache は exp205 v2 と decompressed SHA が一致し、best RMSE 差 `3.8106e-06` は近似一致として許容する。exp072 full cache SHA は exp072 v2 reference と一致せず、full artifact exact parity は未証明。wall time は約 9h29m45s で、exp072 v2 + exp205 v2 の単純合算より遅い。HMM outer parallel option は実装済みだが、今回の完了 run では `outer_workers=1` のまま使っていない。
