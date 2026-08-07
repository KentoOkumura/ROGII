# exp186_typewell_late_range_pfbeam_generation_soft_prior

## 状態

- ルート: pf_beam
- 状態: completed_train_feature_cache_direct_pfbeam_rejected_no_submit
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-04
- 親実験: `typewell_late_range_pfbeam_generation_soft_prior` backlog / `exp072` / `exp176`

## 目的

既存 full replay cache を入力にせず、raw train well/typewell から full replay train feature cache を作り直す。
PF/Beam/likelihood-PF 生成時に、typewell TVT 前半 range へ戻る候補へ soft prior penalty を加える。

## 仮説

late prefix well で typewell TVT 前半 range へ戻る PF/Beam path は、hard 禁止ではなく生成時の soft penalty として扱えば、GR likelihood の自由度を残したまま外れ候補を減らせる可能性がある。

## 変更点

- exp072-style full replay train feature cache を raw train files から再生成する。
- 既存 full replay cache は generation input として読まない。
- `PF_ANCC`、`PF_Z`、Beam path cost、128-seed likelihood-PF に typewell late-range soft prior を入れる。
- selected soft prior は `pct50_strong2_pct70_weak0p5`。
- LightGBM 学習、inference port、submit は作らない。
- v1/v2 の 192-row prefix-holdout audit は superseded 実装として履歴に残す。

## 生成物

- `artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_pixiux_likpf_late_soft_prior_public_replay_train_features.csv.gz`
- `artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_feature_schema.csv`
- `artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_summary.json`

## 実行結果

| 項目 | 値 |
| --- | --- |
| Kaggle kernel | `kentookumura/exp186-typewell-late-soft-pfbeam-train` v3 |
| status | `KernelWorkerStatus.COMPLETE` |
| rows | 3,783,989 |
| wells | 773 |
| columns | 199 |
| feature_count | 196 |
| runtime | 15,783.764 sec |
| feature generation | 14,053.477 sec |
| train features raw SHA | `4bb7a43278ec65143d61c3451353735093995d5258aad665b901237a6a469185` |
| train features decompressed SHA | `b4dd75312d91b21f55b8d1ad09a8590c6bb75857ddfbbbc84d7db175dbb75d15` |

## exp072 direct PF/Beam 比較

full train replay cache の同一 row 3,783,989 件で、exp072 と exp186 の candidate を RMSE TVT で比較した。
真値は `last_known_tvt + target`、Beam / likelihood-PF の `_d` 系 candidate は `last_known_tvt` を足して absolute TVT に戻した。

| candidate | exp072 RMSE | exp186 RMSE | delta |
| --- | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 14.220030 | -0.273031 |
| `pf_z` | 17.788174 | 17.679589 | -0.108585 |
| `beam_mean` | 15.774328 | 15.753703 | -0.020624 |
| `beam_cons` | 16.023008 | 16.025383 | +0.002374 |
| `beam_sm5` | 16.313542 | 16.309361 | -0.004181 |
| `beam_med` | 15.987519 | 15.988469 | +0.000950 |
| `likpf_mean` | 11.594898 | 12.942278 | +1.347381 |

## 検証方針

- Kaggle Notebook 実行を正とする。
- raw train horizontal/typewell 773 wells を入力にする。
- 既存 full replay cache を generation input として読まない。
- schema は exp072 互換の 196 features を期待する。
- gzip integrity、raw gzip SHA、decompressed SHA、row count、schema lines を確認する。

## 所見

v3 で corrected scope の full replay train cache rebuild は完了した。
v1/v2 の prefix-holdout audit は、今回の入力/出力意図と違うため正式結果にしない。

この実験単体は LightGBM CV/LB を持たない。
ただし direct PF/Beam RMSE TVT では、exp072 の主力候補 `likpf_mean` が RMSE +1.347381 と大きく悪化した。
`pf_ancc` と `beam_mean` は小幅改善したが、exp072 cache replacement としては不採用。

## 実行入口

- 学習 notebook: `exp186_typewell_late_range_pfbeam_generation_soft_prior_train.ipynb`
- 推論 notebook: `exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp186_typewell_late_range_pfbeam_generation_soft_prior`
- notebook 実行: Kaggle kernel run を正とする。

## 注意

この実験は train feature cache generation までで、CV/LB は持たない。
改善有無は downstream で既存 exp072 cache と同条件比較して判断する。
test 側 feature は downstream inference notebook で同じ soft-prior generation code から raw test files を再生成する。
