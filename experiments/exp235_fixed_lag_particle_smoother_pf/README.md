# exp235_fixed_lag_particle_smoother_pf

## 状態

train-side audit は完了し、lag64 は不採用です。ユーザー判断により lag128 / lag256 は実行せず、
inference / submission は無効です。

## 仮説

後続 GR を使って過去の particle trajectory を再評価すれば、forward particle mean より mode slip 後の TVT を改善できる可能性があります。

## 検証方針

exp072-compatible forward PF を固定し、lag64 の ancestor-trace posterior mean を4 deterministic
well shardで生成して strict mergeした。3,783,989 rows / 773 wells で exp072 `likpf_mean` と比較した。

## 所見

`pf_lag64_mean` は RMSE 13.495448、exp072 `likpf_mean` は11.594898で、+1.900550悪化した。
within-10も0.673755でcontrolの0.772807を下回ったため、lag64は不採用とする。なおexp235とexp072で
particle seedが一致しないため、これは実装candidateの採否であり、smoothing単独のseed-paired因果推定ではない。

`exp072` likelihood-PF の forward filter を固定し、particle state と ancestor map を bounded ring buffer に保持する fixed-lag smoother の train-side audit です。後続 `64/128/256` rows の同一 well の GR で過去 TVT state を再推定します。

- Route: `pf_beam`
- Parent/control: exp072 likelihood-PF。control cache は再生成しない。
- 実行 variant: `lag64`。`lag128` / `lag256` はユーザー判断で中止
- LightGBM / folds / boosters: `0 / 0 / 0`
- GPU / inference / submission: disabled

正解 TVT は scoring と coverage readout にだけ使い、smoothing の入力には使いません。実装candidateは
train-sideで不採用となり、fixed-lag PF枝は閉鎖した。
