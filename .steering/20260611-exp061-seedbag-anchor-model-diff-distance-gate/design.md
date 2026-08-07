# 設計

## アプローチ

exp059 の train-side model-diff raw candidate はそのまま学習し、postprocess candidate として次の式を追加する。

```text
seedbag_gate_pred = exp054_foldout + alpha(distance_bucket) * (raw_model_diff_pred - exp054_foldout)
```

inference では同じ式を full-train exp054 source prediction に適用する。`exp054_foldout` / full-train exp054 は exp054 config から seed-bagging source model を再学習して生成し、submission artifact を直接読まない。

固定 profile は 3 つに絞る。

- `near_mid_a0p25_far0`: `rows_2500_plus` は exp054 anchor に戻し、それ以外は 25% 補正。
- `near_mid_a0p50_far0`: far は戻し、それ以外は 50% 補正。
- `global_a0p25`: 全 bucket で 25% 補正する比較候補。

## 実験範囲

- 対象実験: `exp061_seedbag_anchor_model_diff_distance_gate`
- Route: `ml_model`
- 親実験: `exp059_pf_model_diff_foldsafe_surface_shrink`
- 変更する変数: exp054 anchor と exp059 raw model-diff の距離別混合 alpha
- 固定する変数: exp059 feature families、LightGBM params、source model generation、PF/Beam selector、visible physical branch

## リスク

- リークリスク: source predictions は audit split ごとに train-fold wells のみで fit する。inference は full train fit を使うが、これは test label を使わない。
- CV/LB 不一致リスク: exp029 pseudo-test surface と Public LB は一致しない可能性がある。exp059 は Public LB で exp054 に +0.022 負けているため、far bucket を守る profile を優先する。
- ランタイム/メモリリスク: exp059 と同じ source model 再学習に加えて postprocess 候補を数個追加するだけなので、実行時間増は小さい。
