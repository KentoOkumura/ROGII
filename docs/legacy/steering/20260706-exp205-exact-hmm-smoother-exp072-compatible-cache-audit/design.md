# 設計

## アプローチ

amerhu notebook の exact HMM smoother を `exact_hmm_smoother.py` に薄く移植し、raw train per-well CSV から unknown suffix rows だけの HMM train feature cache を生成する。HMM 生成は既存 exp072 cache を入力に使わない。比較段階だけ exp072 cache を読み、`id` 厳密一致で HMM / exp072 `likpf_mean` / fixed blend を評価する。

## 実験範囲

- 対象実験: `exp205_exact_hmm_smoother_exp072_compatible_cache_audit`
- Route: `pf_beam`
- 親実験: `exact_hmm_smoother_exp072_compatible_cache_audit` backlog
- 参照実験: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: HMM posterior mean/std/loglik を exp072-compatible train cache として生成し、fixed blend を comparison artifact として評価する。
- 固定する変数: exp072 deterministic cache、train raw files、unknown suffix rows、metric、distance bucket、HMM default config。

## 再現性設計

- seed policy: HMM 本体は乱数なし。optional multi-cut PF baseline は default 無効、使う場合も公開 notebook と同じ fixed seed_base 0 の診断扱い。
- stochastic 処理の有無: default train cache generation は deterministic HMM のみ。
- PF/Beam / likelihood-PF / seed bagging の有無: exp205 の生成対象には含めない。exp072 cache の `likpf_mean` は comparison baseline として読むだけ。
- 並列処理と乱数の関係: `_hmm2_fb` は numba `parallel=True` を使うが乱数を消費しない。floating reduction 差の可能性は deterministic submission anchor ではないため注意として記録する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU を既定にする。amerhu notebook は T4 metadata だが、本実験は train feature cache audit で GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: gzip raw SHA と decompressed content SHA を summary に記録する。test feature は生成しない。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は生成しないため not applicable とする。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` 後に metadata と bootstrap support files を確認する。

## リスク

- リークリスク: HMM generation は raw train の `TVT_input` prefix、GR、Z、typewell だけを使う。unknown suffix の `TVT` は target 列と comparison metric のみに使い、feature source へ入れない。
- CV/LB 不一致リスク: train-side direct comparison であり Public LB 根拠ではない。global RMSE が良くても worst-well、distance bucket、hidden-like subgroup が弱ければ diagnostic として閉じる。
- ランタイム/メモリリスク: default `step=0.35` / `n_rates=41` は full 773 wells で重い可能性がある。full run 前に Kaggle CPU runtime を監視し、exp072 v2 runtime を大きく超える場合は smoke / numba 小修正に留める。
- 再現性リスク: numba parallel と gzip metadata の差が残る。証拠は decompressed content SHA を主にし、deterministic submission anchor とは扱わない。
