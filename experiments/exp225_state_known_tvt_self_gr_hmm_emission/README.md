# exp225_state_known_tvt_self_gr_hmm_emission

## 目的

`state_known_tvt_self_gr_hmm_emission` backlog を実装する。`exp209` exact HMM の typewell GR emission を主軸に残し、同一 horizontal well の known prefix から作る `TVT_input -> GR` 曲線を、HMM candidate state が known-prefix TVT 範囲内にある場合だけ弱い emission boost として足す。

## 状態

- Route: `ensemble`
- 状態: Kaggle train v1 完了 / 不採用
- 新規 LightGBM 学習なし
- exp072 full cache 再生成なし
- raw-test inference / submission なし

## 仮説

exp223 の self-GR motif boost は exp072 `likpf_mean` より改善したが、exp209 HMM/likPF blend には届かず、worst-well regression が大きかった。motif matching surface を全候補 grid に作るのではなく、known prefix の `TVT_input -> GR` 曲線が定義できる TVT state だけで weak boost すれば、self-GR の wrong-depth 吸着を抑えられる可能性がある。

## 実装方針

各 well で HMM の state TVT grid と既存 typewell GR likelihood を作る。追加の self-GR likelihood は、finite `TVT_input` と finite `GR` を持つ known prefix 行だけから TVT 順の曲線を作り、candidate state `grid[j]` が `[known_tvt_min, known_tvt_max]` に入るときだけ計算する。範囲外 state の self-GR boost は 0 のままにする。

HMM emission は次の形にする。

```text
logL_total[row, state] =
  logL_typewell_GR[row, state]
  + alpha * quality[row] * boost_self_GR[row, state]

boost_self_GR[row, state] = clip(centered_logL_self_GR, 0, clip)
```

初回 active variant は `alpha = 0.07`、`clip = 1.0`、mode `boost_only` の 1 通り。model/config/fold/booster count は 0。

## 検証方針

比較対象は exp072 `likpf_mean` / `pf_ancc` / `pf_z` / `beam_mean` と state-known self-GR HMM variant。overall、distance bucket、hidden-like split、by-well regression、HMM std calibration、step-delta rates、self-GR quality / agreement / state-valid rate を見る。

global CV が小改善でも、worst-well、near-row、hidden-like stress、state-valid rate、self-GR disagreement bucket が弱い場合は diagnostic で閉じる。raw-test inference や submit には進めない。

## 所見

実装と Kaggle train v1 は完了。`hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100` は RMSE 14.212954500 で、exp072 `likpf_mean` 11.594897668 から +2.618056832 悪化した。近傍 bucket (`000_050` / `050_100`) は小改善したが、`1000_plus` は +2.931795、hidden-like は +2.84 から +2.94 RMSE 悪化した。

候補 state ごとの known-prefix TVT 範囲 trigger は実装通り動いたが、self-GR 曲線の wrong-depth 吸着を十分に抑えられなかった。追加 grid、raw-test regeneration、inference、submit は行わない。
