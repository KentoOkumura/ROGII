# 設計

## アプローチ

exp209 exact HMM の row-state log score に、group-safe exp226 `tvt_geop` を中心とする
Gaussian unary を加える。

`log p_t(s) = log p_GR(GR_t | TVT_s) - 0.5 * 0.50 * ((TVT_s - tvt_geop_t) / 20)^2`

これは exp226 と HMM の出力平均ではなく、全時点の posterior を joint re-decode する。
exp226 `tvt_pred` は GR 補正済みなので使わず、GR 二重利用を避ける。事前 read-only audit では
exp226 OOF 3,783,989 rows / 773 wells の `tvt_geop` が exp209 固定 grid 内に 100% 入った。

## 実験範囲

- 対象実験: `exp279_exp226_geop_centered_exact_hmm_redecode`
- Route: `pf_beam`
- 科学的親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 実装参照: exp221 Gaussian unary、exp270 self-contained exact HMM、exp226 geometry OOF
- 比較対象: 保存済み exp226、exp209、exp072 likelihood-PF、exp263 fixed formula
- 変更する変数: exp209 GR emission に exp226 geometry Gaussian unary 1点を加える
- 固定する変数: HMM grid / rate states / transition / process noise / GR emission / calibration / missing-GR / score rows
- 実行量: active HMM variant 1、773 well-runs、LightGBM config 0、fold training 0、booster 0
- 除外: control再生成、parameter grid、top-K path、PF、blend/selector、inference、submission

## 再現性設計

- seed policy: RNG なし。well は文字列昇順、exp226 の保存 fold を使用する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済み likelihood-PF は exp263 比較式の再構成だけに使う。
- 並列処理と乱数の関係: 外側 well loop は single process。Numba 4 threads は決定的な forward-backward kernel のみで RNG なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU off、internet off。exp221 実測から約5時間を想定する。
- train cache / test feature regeneration の SHA 記録方針: exp226 / exp209 / exp072 / hidden-like input の raw SHA と gzip decompressed SHA、raw well file SHA、row/schema/content SHA を保存する。
- model manifest / prediction / submission SHA 記録方針: fitted model はないため decoder manifest SHA を model manifest 代替とし、candidate prediction content SHA を保存する。inference/submission SHA は対象外。
- Kaggle package bootstrap 確認方針: push 承認後に canonical private CPU package を prepare し、source / package / bootstrap の config と train source SHA を一致確認する。

## リスク

- リークリスク: exp226 OOF が対象wellを donor/kappa fitから除外した fold predictionであることを SHA と fold-by-well で確認する。true TVT は decoder API に渡さず、candidate freeze 後の readout だけに使用する。
- CV/LB 不一致リスク: exp263 は OOF 8.238331 / Public LB 7.800 で方向は整合したが、CV gainだけで inference化せず全subgroup guardを要求する。
- ランタイム/メモリリスク: 773 exact-HMM well-runs と約2 GBのexp072 gzip読込が支配的。candidateは1本、outer workers 1、well単位解放とする。
- 再現性リスク: Numba parallel reductionの微小差があり得るため、Kaggle rerun前は deterministic anchor と呼ばない。
- 科学的リスク: unary がexp226の誤差へHMMを固定しGR補正を弱める可能性がある。sigma/lambdaの事後救済はせず1点で棄却可能にする。
