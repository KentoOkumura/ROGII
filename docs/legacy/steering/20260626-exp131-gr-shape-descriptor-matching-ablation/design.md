# 設計

## アプローチ

exp072 の train feature cache から既存 PF/Beam 候補を読み、raw train horizontal well の GR と prefix `TVT_input` だけで candidate TVT を prefix GR 位置へ写す。各評価 row の GR window と候補位置の GR window を比較し、複数の target-free score variants を作る。

比較する score variants:

- `raw_point_real`: 評価 row と候補位置の単点 GR 差。
- `ncc_window_real`: local z-score window の NCC。
- `banded_shift_real`: 小さい shift offset を許した local window MAE。
- `shape_descriptor_real`: z-score shape、derivative、curvature、energy、peak/trough proxy、missing gap を合わせた descriptor distance。
- `combo_descriptor_real`: raw / window / banded / shape を混ぜた総合 score。
- `combo_descriptor_shuffled`: GR を well 内 roll した negative control。
- `no_gr_constant`: GR を使わない constant score control。

評価は candidate AUC/logloss、topK coverage、score top1 RMSE、bucket / by-well stress とし、直接 TVT 候補や提出は作らない。

## 実験範囲

- 対象実験: `exp131_gr_shape_descriptor_matching_ablation`
- Route: `pf_beam`
- 親実験: `gr_shape_descriptor_matching_ablation` backlog
- 変更する変数: GR matching cost / score variant
- 固定する変数: exp072 candidate surface、raw train data、candidate set、評価 rows、thresholds

## 再現性設計

- seed policy: no new RNG。shuffled control は random shuffle ではなく deterministic `np.roll`。
- stochastic 処理の有無: exp131 内ではなし。upstream exp072 PF/Beam cache は外部生成物として SHA を記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 再実行しない。exp072 cache を読むだけ。
- 並列処理と乱数の関係: 並列処理なし、global RNG なし。
- CPU/GPU runtime と deterministic flags: CPU-only。GPU 学習なし。
- train cache / test feature regeneration の SHA 記録方針: input cache raw SHA、decompressed SHA、schema SHA、生成 wide cache raw/decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model / submission なし。score variant ごとの top1 TVT proxy SHA だけ summary に記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` で package を作り、metadata と bootstrap config を一致させる。

## リスク

- リークリスク: candidate TVT を prefix TVT へ写す処理で validation true TVT を使わない。true TVT は labels / metrics のみに限定する。
- CV/LB 不一致リスク: train-side diagnostic only なので LB に直接対応させない。positive result でも downstream verifier feature 候補に留める。
- ランタイム/メモリリスク: full DTW path は使わず、fixed window / fixed offsets / banded local shift proxy に制限する。
- 再現性リスク: gzip output は raw SHA と decompressed SHA を分ける。deterministic submission anchor とは呼ばない。
