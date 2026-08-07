# exp223_joint_typewell_self_gr_hmm_likelihood_probe

## 目的

`joint_typewell_self_gr_hmm_likelihood_probe` backlog を実装する。`exp209` exact HMM の typewell GR emission を主軸に残し、同一 horizontal well の visible prefix から作る self-GR motif likelihood を弱い clipped boost として足す。

## 状態

- Kaggle train v1 完了
- Route: `ensemble`
- 新規 LightGBM 学習なし
- exp072 full cache 再生成なし
- raw-test inference / submission なし

## 仮説

typewell GR emission は exp209 で train-side に大きく効いているが、typewell peak が曖昧な row では同一 horizontal well の known prefix GR motif も弱い補助情報になり得る。self-GR を候補値として直接使うと壊れるため、HMM emission の clipped boost に限定すれば wrong-depth 吸着を抑えつつ小改善を反証できる。

## 実装方針

各 well で HMM の state TVT grid を作り、既存の typewell GR likelihood を計算する。そのうえで、finite `TVT_input` prefix の GR window descriptor と評価 row の GR window descriptor を照合し、prefix TVT 周辺に Gaussian mixture の self-GR likelihood surface を作る。

HMM emission は次の形にする。

```text
logL_total = logL_typewell_GR + alpha * quality_self * clip(centered_logL_self_GR, -c, c)
```

初回 active variants は runtime 制限を優先し、`alpha = 0.07 / 0.15`、`clip = 1.0`、mode `boost_only` の 2 通りに絞る。`quality_self` は prefix anchor coverage、GR missing rate、match sharpness、top1-top2 gap、typewell peak との近さから作る。

## 検証方針

比較対象は exp072 `likpf_mean` / `pf_ancc` / `pf_z` / `beam_mean` と self-GR HMM variants。overall、distance bucket、hidden-like split、by-well regression、HMM std calibration、step-delta rates、self-GR quality / agreement bucket を見る。

global CV が小改善でも、worst-well、near-row、hidden-like stress、self-GR disagreement bucket が弱い場合は diagnostic で閉じる。

## 所見

Kaggle train v1 は full train 3,783,989 rows / 773 wells で完了。best は `hmm_selfgr_boost_only_a070_c100` で RMSE 11.349950650、exp072 `likpf_mean` 11.594897668 から -0.244947018 改善した。distance bucket と hidden-like は改善し、runtime も約10h50mで12h以内に収まった。

一方、exp209 HMM/likPF blend RMSE 10.269696 には届かず、最大 by-well regression は `b19b0395` +46.954683 RMSE と大きい。したがって raw-test regeneration / inference / submit には進めず、後続では direct candidate や replacement ではなく confidence feature / regression guard 材料に限定する。
