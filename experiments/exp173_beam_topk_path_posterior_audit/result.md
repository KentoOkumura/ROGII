# exp173_beam_topk_path_posterior_audit 結果

## 仮説

Beam search 本体の retained top-K path と cost から作る posterior 候補は、GR shift-scan proxy よりも Beam の mode ambiguity を直接反映し、候補生成や confidence diagnostic の材料になる可能性がある。

## 設定

- 親: exp072_exp063_full_replay_feature_cache
- 検証: train_well_pseudotail_beam_topk_path_cost_audit
- メトリック: RMSE
- シード: 42
- Beam variants: 3
- LightGBM boosters: 0

## 結果

| メトリック | 値 |
| --- | --- |
| baseline `likpf_mean` RMSE | 11.594897672 |
| best posterior RMSE | 15.972927962 |
| best posterior delta vs `likpf_mean` | +4.378030290 |
| best top-K oracle RMSE | 15.549454381 |
| best top-K oracle delta vs `likpf_mean` | +3.954556709 |
| Public LB | なし |
| Private LB | なし |

## 再現性

- deterministic anchor: false
- seed policy: no_new_rng_beam_dynamic_programming
- kernel version: `kentookumura/exp173-beam-topk-path-posterior-audit-train` v2
- feature content SHA: source decompressed `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- generated SHA: candidate metrics `87126c3131a861616957732f9e3c3a57a32c526418bef072300e5453928d7e99`、top-K paths decompressed `cf23b20a5b2ee9c8266f6272374463ec49cf8229c78570b73908cd346f4c73cc`
- model SHA / manifest SHA: model なし
- prediction SHA: prediction なし
- submission SHA: submission なし
- rerun result: v1 は kernelspec metadata 不足で Papermill 起動前に失敗。v2 で完了。

## 解釈

Kaggle train-side audit は negative。best posterior は `beam_topk_sm11_bw64_posterior_mean_t16` で RMSE 15.972927962 / MAE 10.852413073 / within10 0.602145249。primary baseline `likpf_mean` RMSE 11.594897672 から +4.378030290 悪化した。top-K oracle でも RMSE 15.549454381 で `likpf_mean` から +3.954556709 悪く、Beam retained top-K 内に baseline を超える headroom は見えない。

したがって Beam top-K posterior mean、top2 commit、top-K weighted mean、top-K oracle を direct candidate、PF/Beam likelihood 変更、inference port、submit へ進めない。posterior mean は物理的に無効な中間 trajectory になり得るという懸念以前に、train-side pseudo-tail で `likpf_mean` に大きく届かない。

## 次

この backlog は完了/不採用として閉じる。後続は Beam top-K posterior ではなく、既に positive な exp157/exp158 selector surface や typewell late-range prior など、候補選択・confidence 側を優先する。
