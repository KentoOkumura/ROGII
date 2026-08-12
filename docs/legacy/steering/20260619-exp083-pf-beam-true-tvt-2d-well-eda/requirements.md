# 要件

## 依頼

`pf_beam_true_tvt_2d_well_eda` を実装する。現 anchor の `exp073_gpu_reproducibility_guard_for_exp063_full_replay` が入力として使う `exp072_exp063_full_replay_feature_cache` の full replay train feature cache から、PF/Beam 系候補と true TVT を well ごとに 2D 可視化する。

## 制約

- Route: `pf_beam`
- 親入力: `exp072_exp063_full_replay_feature_cache`
- Anchor: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- `target` は `TVT - last_known_tvt` なので、描画時は `true_tvt = last_known_tvt + target` に戻す。
- `*_d` 候補は `last_known_tvt + delta` として TVT 空間に戻して比較する。
- true TVT は EDA と評価だけに使い、PF/Beam 生成、候補選択、提出ルールには使わない。
- Kaggle CPU notebook で exp072 kernel source を mount して実行する。
- PNG は notebook output の軽量生成物として扱い、代表 plot と manifest を優先する。

## 受け入れ基準

- exp072 feature cache から source artifact を探索できる。
- well 単位で true TVT、PF ANCC、Beam mean、likelihood-PF、hybrid、主要候補を重ねた PNG を保存する。
- 全 well summary と plot manifest を CSV として保存する。
- source gzip は raw SHA と decompressed content SHA を summary に記録する。
- `validate_experiment.py` と Kaggle notebook prepare が通る。
