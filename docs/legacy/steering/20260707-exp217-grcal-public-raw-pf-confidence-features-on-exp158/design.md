# 設計

## アプローチ

exp158 の selector training / Viterbi flow を維持し、exp214 の public-like raw PF diagnostics を full train で再生成して `pubraw_` confidence feature として追加する。

## 実験範囲

- 対象実験: `exp217_grcal_public_raw_pf_confidence_features_on_exp158`
- Route: `pf_beam`
- 親実験: `exp158_segment_continuity_selector_on_exp157`
- 変更する変数: candidate scorer の入力特徴量に `pubraw_` row/candidate-long feature を追加
- 固定する変数: selectable candidates、fold、LightGBM head 数、Viterbi grid、direct prediction policy

## 再現性設計

- seed policy: exp214 helper と同じ stable SHA256 per query well / seed index
- stochastic 処理の有無: public-like likelihood-PF particle propagation / resampling、LightGBM training、candidate-long subsampling
- PF/Beam / likelihood-PF / seed bagging の有無: public raw PF 500 particles x 128 seeds / well
- 並列処理と乱数の関係: PF helper は per-well seed を使い、parallel RNG は使わない
- CPU/GPU runtime: v1 は CPU only で cancelled_no_cv。v2 は Kaggle T4 GPU runtime でも cancelled_no_cv。`pubraw_` PF generation は CPU/Numba のままなので、v3 方針は CPU cache stage と selector train stage を分離する。internet disabled。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする
- model manifest / prediction / submission SHA 記録方針: model manifest SHA、OOF prediction decompressed SHA、variant prediction SHA を summary に記録する。submission は生成しない
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` と `make validate-exp` を通す

## リスク

- リークリスク: exp214 scoped output join、true TVT / oracle / true-error rank の feature 化を避ける
- CV/LB 不一致リスク: train-side selector audit なので、positive でも raw-test parity と hidden-like stress までは submit しない
- ランタイム/メモリリスク: full train public raw PF regeneration が重い。`pubraw_` cache を先に作り、selector train は cache を読む。candidate-long model は exp183/184 と同じ 120k/fold cap と chunked OOF prediction にする
- 再現性リスク: PF は stochastic だが stable per-well seed で固定する。cache stage と selector train stage に分けても、モデル・候補・fold・LightGBM parameters は v1 から変えない
