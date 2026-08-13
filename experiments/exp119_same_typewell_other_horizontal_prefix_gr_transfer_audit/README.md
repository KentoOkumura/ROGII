# exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit

## 状態

- ルート: ensemble
- 状態: completed_train_side_rejected_no_submit
- CV: best `likpf_mean` RMSE 11.594897672
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-24
- 親実験: `exp109_typewell_neighbor_prior_features`
- 備考: 当初 `exp118` として作成したが、既存 `exp118_spatial_neighbor_prior_confidence_gate_on_exp092` と衝突したため `exp119` に改番した。Kaggle v1/v2 は旧 kernel id `kentookumura/exp118-same-typewell-prefix-gr-transfer-train` の履歴として残す。

## 仮説

同じ native typewell overlap group に属する他 horizontal well の visible prefix GR と `TVT_input` は、query well の evaluation-zone GR に対応する局所 TVT path の弱い prior になる可能性がある。

exp109 は同じ group の source tail true TVT drift を `md_since` 軸で使ったが、この実験では source tail を使わず、source 側は疑似 predict start 前の raw GR と `TVT_input` 相当だけに制限する。

## 変更点

- exp065 の `common_typewell_cluster_assignments.csv` から `native_overlap_0p999` group を読む。
- exp099 v2 train feature cache の pseudo-tail row id から、各 well の pseudo predict start と anchor を復元する。
- validation well の evaluation-zone raw GR window を、train-fold source well の prefix raw GR window と照合する。
- source prefix `TVT_input` から offset / local slope / local path delta を作り、`likpf_mean` / `pf_ancc` / `beam_mean` への clipped correction も評価する。
- same-typewell deterministic random control と different-typewell GR-match control を同時に保存する。

## 検証方針

- Fold: fixed seed の well-grouped 5 folds
- Group: `well`
- 主比較: `likpf_mean`、same-typewell GR transfer、same-typewell random control、different-typewell GR-match control
- 指標: RMSE / MAE / within10 / match coverage / match score / source count / distance bucket / worst-well regression / signal correlation
- Leakage Check: source は train-fold wells の pseudo-prefix のみ。valid well と同 fold valid true TVT は source に入れない。

## 実行入口

- 学習 notebook: `exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit_train.ipynb`
- 推論 notebook: `exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は明示的な smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | best `likpf_mean` RMSE 11.594897672 |
| Public LB | - |
| Private LB | - |

Kaggle train v1 は約 2088 秒で `DeadKernelError: Kernel died`。v2 で source / candidate grid を縮小して完了した。

v2 の best は baseline `likpf_mean` のまま。same-typewell GR transfer の最良は `native_overlap_0p999_same_typewell_gr_match_slope_likpf_mean_corr_a0p1_c20` で RMSE 11.614959308、`likpf_mean` から +0.020061636 悪化した。different-typewell control の最良 RMSE 11.607097336、same-typewell random control の最良 RMSE 11.609221461 よりも弱い。

## 所見

### 良かった点

- v2 では 3,783,989 rows / 773 wells の full train pseudo-tail audit が完走した。
- negative controls と同一枠で same-typewell GR transfer を比較できた。

### 悪かった点

- baseline `likpf_mean` を超える候補はなかった。
- same-typewell GR match は random / different-typewell control より弱く、期待した typewell 固有の GR transfer signal は確認できなかった。

### リスク / 注意

- raw GR 波形の直接転用は過去 exp008 / exp017 / exp042 で悪化履歴があるため、global OOF 改善だけでは提出候補にしない。
- test batch 内の他 well `TVT_input` 利用は rules / leakage 解釈リスクがあるため、この実験では扱わない。
- source prefix だけを使うため、coverage や match score が高くても direct TVT replacement としては弱い可能性がある。

## 次

この方針は閉じる。raw GR transfer を hard correction / candidate path として使うのは見送る。使う場合でも GR match score や source count を quality diagnostic として別実験の補助特徴に限定する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
