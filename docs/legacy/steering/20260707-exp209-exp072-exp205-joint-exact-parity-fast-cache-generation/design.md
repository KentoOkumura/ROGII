# 設計

## アプローチ

exp072 の full Pixiux likelihood-PF public replay cache と、exp205 の exact HMM cache / direct comparison を 1 つの Kaggle train notebook にまとめる。最初の高速化はアルゴリズム変更ではなく、exp072 generation 後に comparison 必要列だけを direct comparison に渡して、exp205 側で行っていた 2GB gzip の再展開を省くことに限定する。

HMM well 外側並列は `feature_cache.hmm.outer_workers` として実装する。ただし既定値は `1` とし、serial baseline で exp205 v2 parity を確認してから `2` 以上を試す。outer parallel を使う場合も joblib は入力順で結果を返すため row order は維持される想定だが、Numba thread pool と重なるため OOM / oversubscription / tiny float drift を必ず監視する。

## 実験範囲

- 対象実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`、`exp205_exact_hmm_smoother_exp072_compatible_cache_audit`
- 変更する変数:
  - exp072 generated DataFrame から comparison 必要列を direct comparison へ in-memory で渡す。
  - HMM generator に optional `outer_workers` を追加する。
  - 同一 notebook 内に exp072 cache、HMM cache、comparison、SHA/parity summary を集約する。
- 固定する変数:
  - exp072 `n_jobs=8`、`pf_seeds=128`、`pf_particles=500`、full 196 features。
  - HMM `step=0.35`、`n_rates=41`、`band_pad=100`、`numba_num_threads=4`、その他 exp205 v2 default。
  - comparison candidate、blend weights、distance bucket、by-well、HMM std calibration、step-delta metric。

## 再現性設計

- seed policy: exp072 v2 の stable SHA256 per-well seed を継承する。HMM default は RNG を使わない。
- stochastic 処理の有無: exp072 PF/Beam/likelihood-PF は stochastic component を含むが stable seed で固定する。HMM には RNG はない。
- PF/Beam / likelihood-PF / seed bagging の有無: exp072 full cache は PF/Beam と likelihood-PF を含む。seed/particle count は変更しない。
- 並列処理と乱数の関係: exp072 は per-well stable seed により thread scheduling 依存の乱数消費を避ける。HMM outer parallel は RNG 無しだが Numba thread と重なるため既定無効。
- CPU/GPU runtime と deterministic flags: CPU-only Kaggle notebook。GPU は使わない。
- train cache / test feature regeneration の SHA 記録方針: train cache の raw gzip SHA と decompressed SHA を分ける。reference artifact がある場合は reference SHA と比較する。
- model manifest / prediction / submission SHA 記録方針: モデル、prediction、submission は作らないため対象外。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` 後に generated notebook / metadata / embedded config を検証する。

## リスク

- リークリスク: train-only cache audit で raw test、submission、oracle tuning を使わないため低い。reference artifacts は parity check のみに使う。
- CV/LB 不一致リスク: LB は見ない。direct comparison は train-side diagnostic であり submit 判断には使わない。
- ランタイム/メモリリスク: exp072 full cache DataFrame を in-memory 保持するため peak RAM は高い。HMM outer parallel を有効化するとさらに RAM と CPU oversubscription が増える。
- 再現性リスク: exp072 gzip raw SHA は gzip metadata で変わり得るため decompressed SHA を主証拠にする。HMM Numba parallel は tiny float drift の可能性があるため serial baseline を先に確認する。
