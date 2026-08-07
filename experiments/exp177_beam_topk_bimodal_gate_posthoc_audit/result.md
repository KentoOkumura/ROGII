# exp177_beam_topk_bimodal_gate_posthoc_audit 結果

## 仮説

exp173 の Beam top-K posterior は全 row では `likpf_mean` に大きく負けたが、`top1_top2_sep`、`top2_cost_gap_per_row`、`topk_entropy`、`topk_spread` が示す二峰性 / ambiguity 条件がある row だけに限定すれば、posterior / top2 / weighted mean が局所的に改善する可能性がある。

## 設定

- 親: exp173_beam_topk_path_posterior_audit
- 検証: fixed_exp173_topk_posthoc_gate_audit
- メトリック: RMSE
- シード: 42
- Beam 再生成: 0
- LightGBM boosters: 0

## 結果

| メトリック | 値 |
| --- | --- |
| baseline `likpf_mean` RMSE | 11.594897884 |
| best gated policy RMSE | 11.837783911 |
| best gated delta vs baseline | +0.242886027 |
| changed subset baseline RMSE | 10.269740849 |
| changed subset candidate RMSE | 12.706185740 |
| max well regression vs baseline | +22.519192863 |
| Public LB | なし |
| Private LB | なし |

## 再現性

- deterministic anchor: false
- seed policy: deterministic_posthoc_grid_no_rng
- kernel version: `kentookumura/exp177-beam-topk-bimodal-gate-posthoc-audit-train` v1
- feature content SHA: exp173 `candidate_wide` decompressed `f993aaed3f59a39f3e367e1c18b3a7a394a254db09c1a5277d90d605621613bd`、`topk_diagnostics` decompressed `08b1ed91742e4352b732fb739fdd59a8b4c53f53582f8a1c295a7b123e070301`
- model SHA / manifest SHA: model なし
- prediction SHA: submission candidate なし
- submission SHA: submission なし
- rerun result: v1 完了。実行時間 409.031 sec。

## 解釈

Kaggle train v1 は negative。最良 policy は `beam_topk_sm11_bw64__and_sep_ge_q90__cost_le_q10__replace_posterior_mean_t1` だったが、`likpf_mean` baseline RMSE 11.594897884 に対して 11.837783911 で +0.242886027 悪化した。Gate は 384,720 rows、実際に 384,695 rows / 71 wells を変えたが、changed subset は RMSE 10.269740849 -> 12.706185740 と +2.436444890 悪化した。

near `000_050` も +0.027170813、longtail `1000_plus` も +0.241960608、Beam-likPF gap top quartile も +0.501165946 悪化し、max well regression は +22.519192863 と guard 0.25 を大きく超えた。したがって二峰性 / low-cost-gap 条件で絞っても、exp173 Beam top-K posterior / top2 / weighted mean を direct replacement、confidence feature 化、inference port、submit へ進めない。

## 次

この backlog は完了/不採用として閉じる。Beam top-K posterior 系の追加再生成 follow-up は行わず、PF/Beam route の後続は exp157/158 の selector/continuity surface や、exp178/179 の learned GR match signal のような別系統の confidence feature に限定する。
