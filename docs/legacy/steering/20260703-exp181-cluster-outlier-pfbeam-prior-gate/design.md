# 設計

## アプローチ

exp175 の cluster-outlier gate 実装を親として使い、入力 source だけを ML output から PF/Beam/likPF OOF 候補へ戻す。base candidate は exp109 OOF に含まれる `likpf_mean`、`pf_ancc`、`beam_mean`。prior は exp109 typewell native overlap と exp114 spatial neighbor prior を使う。

補正式は exp109 / exp114 と同じ `base + alpha * clip(prior - base)`。cluster gate、prior quality gate、alpha、clip の grid を評価し、exp109 / exp114 の global best を reference policy として追加する。

## 実験範囲

- 対象実験: `exp181_cluster_outlier_pfbeam_prior_gate`
- Route: `pf_beam`
- 親実験: `exp109_typewell_neighbor_prior_features`
- 比較親: `exp114_spatial_neighbor_prior_signal_audit`
- gate 親: `exp175_cluster_outlier_typewell_prior_gate`
- 変更する変数: correction target を PF/Beam/likPF OOF 候補へ変更し、cluster-outlier gate を prior correction に追加する。
- 固定する変数: upstream OOF candidates、typewell/spatial prior、cluster assignment、fold-safe validation rows、alpha/clip grid。

## 再現性設計

- seed policy: exp181 は deterministic posthoc grid で乱数なし。upstream output は固定ファイルとして SHA を記録する。
- stochastic 処理の有無: exp181 自体にはなし。
- PF/Beam / likelihood-PF / seed bagging の有無: 再生成なし。固定 exp099/exp109 OOF candidate を読む。
- 並列処理と乱数の関係: 乱数なし。grid score は deterministic。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled。LightGBM / booster 学習なし。
- train cache / test feature regeneration の SHA 記録方針: exp109 OOF、exp114 OOF、exp065 cluster assignment、exp114 geometry、exp115 roles の SHA を summary に記録する。gzip は decompressed content SHA も記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest なし。top gated prediction gzip は raw / decompressed SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に validate し、config と notebook bootstrap を同じ source から生成する。

## リスク

- リークリスク: gate と prior source へ true TVT を入れない。true TVT は score のみに使う。
- CV/LB 不一致リスク: train-side OOF audit なので、positive でも raw-test/full-train parity までは submit しない。
- ランタイム/メモリリスク: 3 base candidates を縦持ち評価するため、exp175 より行数が増える。PF/Beam 再生成や ML 学習はない。
- 再現性リスク: upstream PF/Beam candidate は deterministic anchor ではないため、この実験単体を submission anchor と呼ばない。
