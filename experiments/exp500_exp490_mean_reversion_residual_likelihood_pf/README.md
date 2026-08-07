# exp500_exp490_mean_reversion_residual_likelihood_pf

## 状態

- ルート: `pf_beam`
- 状態: `stage1_fail_closed_under_override`
- Stage 0: fixed44 technical 13/13 PASS、mechanism safety 3件FAILを維持
- Stage 1: full OOF technical 18/18 PASS、scientific tail guard 2件FAIL
- CV: `8.813504627`、Public LB / Private LB / Submit ID: なし / なし / なし
- Stage 1 merge: `kentookumura/exp500-mean-revert-resid-likpf-full-merge` version 3、id_no `129465486`
- inference / submission: 無効
- 作成日: 2026-08-01、Stage 1完了日: 2026-08-02

## 仮説と変更点

exp486 residual likelihood-PFのoffset / offset-rate遷移中心へ、exp490と同じ
K16区間1つのhalf-life平均回帰`rho_t`だけを追加した。

```text
rho_t = 2 ** (-dMD_t / destination_K16_segment_MD_span)
q_t = 0.998 * rho_t * q_(t-1) + 0.002 * Normal(0, 1)
delta_t = rho_t * delta_(t-1) + q_t * dMD_t + 0.005 * Normal(0, 1)
TVT_t = exp226_tvt_geop_t + delta_t
```

500 particles、128 seeds、noise、roughening、resampling、Gaussian GR emission、
temperature-5 aggregationはexp486から固定した。adaptive gate、parameter grid、
blend / selectorは使っていない。

## Full OOF結果

| prediction | RMSE (ft) |
| --- | ---: |
| exp500 candidate | 8.813505 |
| exp226 final | 9.427110 |
| exp404 likelihood-PF | 10.914522 |
| exp486 residual-PF | 11.139812 |

exp404比`2.101017 ft`改善し、5/5 foldsと全固定scopeを改善した。一方、by-well p95は
`+6.653601 ft`、worst wellは`+46.154671 ft`悪化したため、全AND gateをFAILした。

## 検証方針

well単位の固定5 foldsを使い、candidateと保存exp404 / exp486 / exp226を同じunknown suffix rowsで
比較した。4 shardのpredictionと診断SHAを結合・freezeしてからtruth、control、fold、scope、episodeを
attachした。pooled、fold、固定6 scope、episode、by-well p95、worst-wellを全AND判定した。

## 実行量

- 4 deterministic shards、773 wells、3,783,989 rows
- 98,944 seed-well trajectories、49,472,000 particle starts
- control PF / HMM / Beam / LightGBM / booster / GPU再実行: 0
- 最大shard wall `8,465.69 sec`、merge/evaluation `956.79 sec`

## 実行入口と再現性

- 正規train Notebook: `exp500_exp490_mean_reversion_residual_likelihood_pf_train.ipynb`
- Jupytext source: `exp500_exp490_mean_reversion_residual_likelihood_pf_compact_selfcontained_train.py`
- contract test: `test_exp500_contract.py`（10件PASS）
- prediction logical SHA: `a4bfa0c48203566be31cfefa4c255182c0bec5949056d6ae688b5252b965210a`
- deterministic anchor: いいえ

`config.yaml`の実行フラグは全てfalseで再実行をdisarmしている。詳細なfold / scope / gate / SHAは
`result.md`と`metrics.json`を参照する。

## 結論

平均RMSEは明確に改善したが、事前登録したwell-tail安全性を満たさなかった。
Stage 0 FAILも保持し、same-OOF rescue、inference、submissionなしで終端閉鎖する。

## 所見

固定K16平均回帰は広い平均指標では有効だが、少数wellの大幅悪化を抑えられない。
tail原因を追う場合も保存artifactだけのreadoutに限定し、このOOFで適用ruleを選ばない。
