# exp109_typewell_neighbor_prior_features

## 状態

- ルート: ensemble
- 状態: completed_train_side_audit_supported_no_submit
- CV: best neighbor correction RMSE 11.143359521
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-22
- 親実験: -

## 仮説

同じ native typewell overlap group に属する train wells の pseudo-tail TVT drift は、query well の `likpf_mean` / `pf_ancc` / `beam_mean` の誤差方向を弱く説明する可能性がある。まず PF 内部の likelihood を変えず、fold-safe な neighbor prior を後段 correction として評価する。

## 変更点

- exp065 の `common_typewell_cluster_assignments.csv` を読み、`native_overlap=1` / `native_overlap=0.999` / `exact_hash` group を使う。
- exp099 v2 train feature cache の `md_since` と `true_tvt - last_known_tvt` を train-fold neighbor curve として使う。
- validation well は query としてのみ扱い、同 fold valid の true TVT は prior source に入れない。
- neighbor prior を `likpf_mean` / `pf_ancc` / `beam_mean` へ clipped correction として掛け、RMSE / MAE / within10 / bucket / by-well で比較する。

## 検証方針

- Fold: fixed seed の well-grouped 5 folds
- Group: `well`
- Stratification: なし
- Leakage Check: neighbor source は train-fold wells のみ。valid well の true TVT は scoring のみに使う。

## 実行入口

- 学習 notebook: `exp109_typewell_neighbor_prior_features_train.ipynb`
- 推論 notebook: `exp109_typewell_neighbor_prior_features_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp109_typewell_neighbor_prior_features`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 11.143359521 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- `native_overlap_0p999_likpf_mean_corr_a0p2_c40` が `likpf_mean` RMSE 11.594897672 から 11.143359521 へ -0.451538151 改善した。
- within10 も 0.772807479 から 0.779883345 へ改善した。
- 全 distance bucket で `likpf_mean` より RMSE 改善した。

### 悪かった点

- well 単位では 413 wells 改善 / 345 wells 悪化 / 15 wells 同値で、最大悪化は +6.594183 RMSE。
- 実装は後段補正であり、PF 内部 likelihood への組み込みではない。

### リスク / 注意

- 本番 inference では full train を neighbor source にできるため、train OOF と coverage が変わる。
- 改善しても raw-test parity audit なしでは submit しない。
- typewell group は TVT の答えそのものではなく、drift prior の補助情報としてのみ扱う。

## 次

- raw-test parity と worst-well gate を別実験で確認する。直接 submit はしない。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
