# exp177_beam_topk_bimodal_gate_posthoc_audit

## 状態

- ルート: PF/Beam
- 状態: completed_train_side_rejected_no_submit
- CV: 11.837783911
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-03
- 親実験: exp173_beam_topk_path_posterior_audit

## 仮説

exp173 の Beam top-K posterior は全 row では `likpf_mean` より大きく悪化したが、二峰性が強い、かつ top2 cost gap が小さい row に限定すれば posterior / top2 / weighted mean が局所的に改善する可能性がある。

## 変更点

- exp173 の保存済み `topk_diagnostics`、`topk_paths`、`candidate_wide`、`candidate_metrics` を読む。
- `likpf_mean` を baseline とし、gate 成立 row だけ exp173 の `top2_commit`、`topk_weighted_mean`、`posterior_mean_t*` に置換する。
- Gate は `top1_top2_sep`、`top2_cost_gap_per_row`、`topk_entropy`、`topk_spread` の target-free quantile と AND 条件だけで作る。
- 追加学習、Beam 再生成、inference port、submission 作成はしない。

## 検証方針

- Fold: なし。固定 upstream train-side pseudo-tail outputs の posthoc audit。
- Group: well 単位の worst-well regression を確認。
- Stratification: near `000_050`、longtail `1000_plus`、Beam-likPF gap top quartile、mode-separation bucket。
- Leakage Check: true TVT は scoring にだけ使い、gate threshold の構築には使わない。

## 実行入口

- 学習 notebook: `exp177_beam_topk_bimodal_gate_posthoc_audit_train.ipynb`
- 推論 notebook: `exp177_beam_topk_bimodal_gate_posthoc_audit_inference.ipynb`
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp177_beam_topk_bimodal_gate_posthoc_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train --title 'exp177 beam topk bimodal gate posthoc audit train' --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は明示的な smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| baseline `likpf_mean` RMSE | 11.594897884 |
| best gated policy RMSE | 11.837783911 |
| delta vs baseline | +0.242886027 |
| Public LB | なし |
| Private LB | なし |

## 所見

### 良かった点

- exp173 の重い Beam 再生成をせず、保存済み output の target-free diagnostics だけで gate を監査できた。

### 悪かった点

- 最良 gate でも `likpf_mean` より RMSE +0.242886027 悪化した。
- changed subset は RMSE +2.436444890 悪化し、max well regression は +22.519192863 と大きい。

### リスク / 注意

- direct replacement、confidence feature 化、inference port、submit には進めない。

## 次

- backlog は完了/不採用として閉じる。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
