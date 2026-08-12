# 要件

## 依頼

`cluster_outlier_alt_typewell_pfbeam_audit` backlog を `exp187_cluster_outlier_alt_typewell_pfbeam_audit` として実装する。

cluster 中心から外れた well では、query well 自身の typewell よりも、近傍 well が参照している別 typewell cluster の composite typewell の方が PF/Beam observation model として良い可能性がある。まず train-side audit に限定し、参照 typewell 差し替えだけを検証する。

## 制約

- Route: `pf_beam`
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- 初回は train-side audit のみ。inference port、submission、hard switch、direct replacement は作らない。
- 代表 typewell 選択に validation tail true TVT、oracle best、true-error rank を使わない。
- 既存 `denoised_gr_pfbeam_generation_audit` や `typewell_late_range_pfbeam_generation_soft_prior` とは混ぜず、参照 typewell の差し替えだけを検証する。
- PF/Beam 再生成を含むため、`docs/06_reproducibility.md` に従い stable per-well seed と SHA 記録を実装する。

## 受け入れ基準

- `docs/legacy/steering/20260704-exp187-cluster-outlier-alt-typewell-pfbeam-audit/` に要求、設計、タスクが記録されている。
- `experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/` に `config.yaml`、train/inference notebook、補助実装、記録ファイルが揃っている。
- train notebook は、入力確認、cluster/strategy 選定、PF/Beam 再生成、metrics/生成物保存をセル上で追える。
- audit は `own_typewell`、`nearest_other_cluster_composite`、`nearby_majority_cluster_composite_k8` を同 seed / 同 particles / 同 beam config で比較できる。
- score rows は exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows` とし、true TVT は `last_known_tvt + target` を scoring のみに使う。
- alt cluster の typewell は representative well 1本ではなく、source cluster に属する available member typewell を TVT bin で結合した 1本の composite typewell として参照する。
- 出力に単体 RMSE、own-vs-alt delta、cluster-outlier subset、distance bucket、path jump、PF diagnostics、worst-well regression、alt selected well summary が含まれる。
- gzip 生成物は raw gzip SHA と decompressed content SHA を分けて記録する。
- deterministic submission anchor として扱わず、`metrics.json` と `SESSION_NOTES.md` に train-side diagnostic only と明記する。
