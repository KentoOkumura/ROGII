# 設計

## アプローチ

`exp226` の数値出力ではなく、区間分割と空間近傍補間という構造だけを再利用する。状態量を
`U = TVT + Z`、MD 方向の rate を `r = dU / dMD` とする。

outer fold ごとに、outer-train の各 donor well の全水平坑跡を MD 長で K=16 の等区間へ分ける。
各区間内で `U`、`X`、`Y` を MD に対して切片付き最小二乗し、次を保存する。

- 区間中心 `(x_j, y_j)`
- 真の donor rate `b_j = dU / dMD`
- 水平進行方向 `h_j = (dX / dMD, dY / dMD)`
- donor well id、outer fold、MD span、finite row 数

target well の未知 suffix も MD 長で K=16 の等区間へ分ける。target 区間中心ごとに、各 donor well
から XY 距離が最小の区間を 1 個だけ残し、その中の近傍 50 wells を使う。距離 `d_j` に対する重みは
`w_j = exp(-0.5 * (d_j / 500 ft)^2)` とする。局所地層勾配 `g_s = (g_x, g_y)` は、

`b_j = h_j dot g_s + error_j`

の weighted ridge として求める。ridge は
`lambda_s = 1e-3 * max(trace(H' W H) / 2, 1e-12)`、切片なしで固定する。target 区間の
rate-prior mean は `mu_s = h_target,s dot g_s` とする。最低 10 donor wells、
effective donor count `(sum w)^2 / sum(w^2) >= 10`、最近傍距離 `<=1500 ft`、
target 方向情報 `h_t' H' W H h_t / (||h_t||^2 sum w) >= 0.30` を全て要求する。
非 finite、target 水平速度ノルム `<0.30`、または `|mu_s - r_prefix| > 0.10` の区間も不成立とする。
不成立区間は target の既知 prefix 末尾 30 step の median `dU/dMD` である `r_prefix` へ fail closed
する。

16 区間中心の `mu_s` を未知 suffix の MD 上で線形補間し、先頭・末尾の外側は端値を保持して
rowwise `mu_t` を作る。これを HMM の absolute TVT unary にはせず、rate transition の平均だけに入れる。
latent residual rate を `q_t = r_t - mu_t` とし、

`q_t = 0.998 * q_(t-1) + epsilon_t`

`U_t = U_(t-1) + (mu_t + q_t) * delta_MD_t + eta_t`

`TVT_t = U_t - Z_t`

とする。`epsilon_t`、`eta_t`、GR/typewell emission、position/rate grid、初期 uncertainty、
posterior mean 出力は exp209 を固定する。初期 residual rate は `q_0 = r_prefix - mu_0`。
residual-rate grid は 41 点、span `[-0.10, +0.10]` とする。

事前 readout は置かない。将来実装後の最初の Kaggle scientific run で、fold-safe donor field の生成から
773 wells の exact HMM までを 1 variant として直結して評価する。

## 実験範囲

- 対象実験: `exp362_segment_local_donor_slope_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 概念参照: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 負の区別対象: `exp355_exp226_dip_rate_prior_on_exp209`。exp355 は exp226 の生成値を利用するため、本実験の入力にも親にも使わない。
- 変更する変数: exp209 の constant rate-prior mean だけを、raw donor truth と target raw geometry から作る K16 segment-local gradient schedule に変更する。
- 固定する変数:
  - emission `gauss`、`sigma_mode=std`、`sig_r=0.002`、`sig_p=0.02`
  - `step=0.35`、`n_rates=41`、`rate_span=0.10`
  - `momentum=0.998`、`start_sig=0.75`、`r0_sig=0.01`、`band_pad=100`
  - target prefix rate の求め方、typewell/GR 観測、posterior mean 出力
  - 5-fold well split、distance / hidden-like / by-well 評価
- scientific variant: `k16_segment_local_gradient_residual_hmm` の 1 個だけ。
- 計算量: 5 reporting folds、outer-valid 各 well 1 回、合計 773 HMM well-runs、model/config/trained fold/booster/GPU は全て 0。
- control: exp209 保存 HMM cache の decompressed SHA
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5` と
  direct RMSE `11.938287417 ft`。control 再実行はしない。
- 禁止: exp226 artifact 読み込み、exp226 path/予測/補正、Stage 0、K/近傍/bandwidth/ridge/HMM grid、
  sigma、momentum の探索、prediction blend、ML selector、inference、submission。

## 再現性設計

- seed policy: 乱数は使わず、well、fold、segment、donor distance、donor well id、row の安定 sort を固定する。5-fold は raw well id の `SHA256("42|well_id")` 順 round-robin とする。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。exact HMM の deterministic dynamic programming のみ。
- 並列処理と乱数の関係: RNG はない。Numba/thread reduction の微小差を避けるため `num_workers=2`、
  `numba_num_threads=2` を固定し、保存順は well id 順へ戻す。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、internet off、GPU off、8.5 h 上限。
- train cache / test feature regeneration の SHA 記録方針:
  - raw well identity と fold assignment の SHA
  - donor segment ledger、target segment geometry、rowwise prior schedule の schema / logical content SHA
  - gzip は decompressed content SHA を主証拠にする
  - 将来 inference を同一 exp で追加する場合は full-train donor field と raw-test schedule を raw から再生成し、row/well/schema/content SHA を別に記録する
- model manifest / prediction / submission SHA 記録方針: 学習 model はない。OOF HMM prediction content SHA と
  Kaggle kernel id/version/source SHA を必須にする。推論・提出を行わない現段階では submission SHA は非該当。
- Kaggle package bootstrap 確認方針: 実装後に `prepare-kaggle-notebooks` を再生成し、埋め込み
  `config.yaml` の variant、CPU/internet、input source、実行 flag が正本と一致することを push 前に確認する。

## リスク

- リークリスク: donor truth を使えるのは outer-train wells だけである。同じ outer-valid fold の全 wells を除外し、
  donor ledger と schedule の SHA を freeze してから unknown-suffix truth を join する。
- CV/LB 不一致リスク: spatial donor support と軌跡方向の分布が train/test で異なる可能性がある。
  donor distance、effective count、directional information、fallback fraction を fold / distance / hidden-like で記録する。
- モデルリスク: scalar path slope から 2D gradient を復元するため、井戸方向が揃う場所では識別性が低い。
  target 方向情報 gate と prefix fallback で fail closed し、同一 OOF 上で ridge や閾値を救済しない。
- HMM リスク: prior mean が強すぎると GR evidence に反して drift する。exp209 の transition variance、
  position noise、momentum、residual span を変更せず、効果を rate mean の 1 変数に限定する。
- ランタイム/メモリリスク: 5 fold donor ledger 再構築と 773 exact HMM runs が主コスト。chunked ledger、
  fold 単位解放、8.5 h fail-closed を設計条件にする。
- 再現性リスク: equal-distance tie、BLAS/Numba reduction、gzip metadata。安定 tie-break、固定 thread 数、
  logical/decompressed content SHA で監査する。
