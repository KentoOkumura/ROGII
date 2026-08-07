# exp205_exact_hmm_smoother_exp072_compatible_cache_audit

## 状態

- ルート: pf_beam
- 状態: completed_train_feature_cache_direct_pfbeam_supported_no_submit
- CV: 10.269699957 (`blend_likpf_hmm_w500`)
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-06
- 親実験: `exact_hmm_smoother_exp072_compatible_cache_audit` backlog

## 仮説

amerhu 公開 notebook の exact HMM smoother は、exp072 の `likpf_mean` と失敗 well が異なる可能性がある。まず train-only の exp072 互換 cache として生成し、HMM 単体、exp072 `likpf_mean`、固定 blend を同じ rows で比較する。

## 変更点

- raw train horizontal/typewell から `hmm_mean_tvt`、`hmm_std`、`hmm_loglik` を生成する。
- exp072 cache と `id` 厳密一致で direct comparison を行う。
- `blend_likpf_hmm_w025/w050/w075` は comparison artifact としてのみ作り、提出候補にはしない。

## 検証方針

- Fold: なし
- Group: `well`
- Stratification: なし
- Leakage Check: unknown suffix の `TVT` は target と metric のみに使い、HMM feature generation には使わない。

## 実行入口

- 学習 notebook: `exp205_exact_hmm_smoother_exp072_compatible_cache_audit_train.ipynb`
- 推論 notebook: `exp205_exact_hmm_smoother_exp072_compatible_cache_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp205_exact_hmm_smoother_exp072_compatible_cache_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 10.269699957 (`blend_likpf_hmm_w500`) |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- Kaggle train v2 で 773 wells / 3,783,989 rows の HMM cache generation が完走した。
- exp072 cache との `id` mismatch は 0。
- `blend_likpf_hmm_w500` は `exp072_likpf_mean` から RMSE -1.325198、MAE -0.668419、within10 +0.020400 改善した。
- 全距離 bucket で `exp072_likpf_mean` より良い。

### 悪かった点

- HMM 単体は RMSE 11.938297 で `exp072_likpf_mean` 11.594898 より弱い。
- best blend でも worst-well regression `b19b0395` +23.036816 RMSE が残る。
- raw-test-compatible HMM regeneration と hidden-like stress は未確認。

### リスク / 注意

- HMM path、fixed blend、posterior std は train-side diagnostic。direct replacement、postprocess blend、PF weight replacement、submit には使わない。
- exp072 full gzip はローカル artifact に常設されていないため、Kaggle comparison では exp072 notebook output を input として参照する。

## 次

- `exact_hmm_likpf_blend_raw_test_port_guard` として raw-test-safe port / hidden-like guard を別 backlog で確認する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
