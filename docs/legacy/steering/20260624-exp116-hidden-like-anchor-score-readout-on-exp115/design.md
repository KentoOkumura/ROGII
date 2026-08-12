# 設計

## アプローチ

exp115 の保存済み Kaggle output から well 単位の hidden-like role と metadata を読み、既存 anchor の OOF / train-side prediction を well id で結合する。row-level prediction がある場合は row ごとの誤差を計算して overall / bucket / by-well を出す。row-level prediction が保存されていない source は by-well metrics を読み、well subset の weighted RMSE と weighted MAE だけを出す。

この実験は「exp115 split で ML を再学習する」ものではない。通常 GroupKFold の OOF や既存 train-side prediction を、exp115 の hidden-like valid wells に限定して読み直す stress readout である。

## 実験範囲

- 対象実験: `exp116_hidden_like_anchor_score_readout_on_exp115`
- Route: `ml_model`
- 親実験: `exp115_hidden_like_spatial_holdout_from_ppt`
- 参照 anchor: `exp092_u_projection_correction_disagreement_fullrun`、`exp073_gpu_reproducibility_guard_for_exp063_full_replay`、`exp098_selector_rank_slot_features_on_exp073`
- 変更する変数: 採点対象 well subset と readout bucket。
- 固定する変数: upstream prediction、upstream fold、モデル重み、exp115 split definition。

## 再現性設計

- seed policy: 新規乱数なし。入力 CSV と config の deterministic merge / groupby のみ。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。参照 prediction に upstream PF/Beam 由来の値が含まれる場合も、この実験では再生成しない。
- 並列処理と乱数の関係: 並列処理なし。pandas groupby の結果は sort key を固定して出力する。
- CPU/GPU runtime と deterministic flags: CPU only。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: exp115 split files と input prediction files の SHA256 を inventory / summary に記録する。gzip は decompressed SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: 新規 model / submission なし。新規 readout CSV / summary の SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` で package 化し、必要な upstream kernel source を metadata に入れる。

## リスク

- リークリスク: exp115 valid true TVT を特徴量や prior source に使わない。true TVT は既存 prediction の評価列としてのみ使う。
- CV/LB 不一致リスク: exp115 は official PPT map 由来の stress holdout であり、exact hidden split ではない。改善/悪化を LB 代替と解釈しない。
- ランタイム/メモリリスク: row-level gzip prediction は大きい。source ごとに読み、結果だけを保持する。欠損 source は skip して inventory に残す。
- 再現性リスク: upstream artifact の取得場所が `/tmp` と Kaggle output で揺れる。複数 path candidates と SHA 記録で実際に採点した入力を明示する。
