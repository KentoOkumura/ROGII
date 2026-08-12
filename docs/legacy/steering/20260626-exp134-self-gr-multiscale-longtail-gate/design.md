# 設計

## アプローチ

exp090 の self-GR multiscale で効いた half-window 25 signal を、
exp092 系の high-drift / PF-dense disagreement regime にだけ効く補助 confidence として posthoc 監査する。
ローカルには exp090 OOF prediction gzip がないため、exp090 artifacts が存在すれば読むが、
存在しない場合でも exp072 full replay cache と raw train horizontal wells から lightweight self-GR signal を再生成する。

評価は LightGBM 再学習ではなく、既存候補/anchor の失敗 regime に self-GR signal が説明力を持つかを見る。
まずは `likpf_mean`、`last_known_tvt`、exp073/exp092 相当の proxy が利用可能な列を同一行で揃え、
target-free gate score と bucket readout を保存する。

## 実験範囲

- 対象実験: `exp134_self_gr_multiscale_longtail_gate`
- Route: `ml_model`
- 親実験: `exp090_lateral_self_gr_match_pseudotail_probe`
- 比較親: `exp092_u_projection_correction_disagreement_fullrun`, `exp126_exp073_exp092_pf_beam_pseudotail_failure_map`, `exp128_trajectory_local_typewell_self_gr_switch_audit`
- 変更する変数: self-GR multiscale signal の gate score / bucket 条件 / diagnostic readout
- 固定する変数: exp072 full replay train feature cache、raw train well files、既存 PF/Beam/likPF candidate columns、true TVT は評価のみ

## 再現性設計

- seed policy: 乱数を使わない。config には seed 42 を記録する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 cache に保存済みの deterministic replay columns だけを読む。
- 並列処理と乱数の関係: self-GR 再生成は deterministic な row/well scan。乱数や thread scheduling 依存を作らない。
- CPU/GPU runtime と deterministic flags: LightGBM 学習なし、GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: 入力 cache、feature schema、再生成 signal CSV の content SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model/submission は生成しない。diagnostic prediction/gate CSV の SHA のみ記録する。
- Kaggle package bootstrap 確認方針: push する場合は train notebook metadata と bootstrap manifest の config/script SHA を確認する。

## リスク

- リークリスク: gate 条件に true TVT、oracle best、absolute error を使わない。評価列としてのみ使う。
- CV/LB 不一致リスク: posthoc train-side audit のため submit 判断には使わない。改善しても raw-test parity と exp115 hidden-like stress が必要。
- ランタイム/メモリリスク: 3.78M rows の full cache と raw well scan が重い。まず vectorized summary と chunk-safe 出力に限定し、モデル学習はしない。
- 再現性リスク: exp090 OOF prediction gzip がローカル未取得のため、必要 signal は再生成可能にする。入力 artifact path と SHA を必ず保存する。
