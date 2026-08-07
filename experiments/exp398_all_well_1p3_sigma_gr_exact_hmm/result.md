# exp398_all_well_1p3_sigma_gr_exact_hmm 結果

## 状態

- `train_side_all_well_sigma_x1p3_gate_failed_closed`
- Kaggle private CPU version 1完了
- decision: `all_well_sigma_x1p3_failed_close_without_rescue`
- inference / submission / version 2 / parameter rescueなし

## 仮説

全wellでexp209のGR scaleを`1.3`倍すると、GR evidenceの過信が弱まりunknown-suffix
RMSEが改善する。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 検証: saved exp209 controlとのpaired 5-fold train-side audit
- 候補: `sigma_eff = 1.3 * clip(sigma_exp209_raw, 10, 60)`、全well、再clipなし
- HMM: exp209 absolute-TVT / capped Gaussian / 41 rates / posterior meanを固定
- 実行量: 1 variant / 773 HMM runs / 5 reporting folds / control rerun 0

## 結果

| メトリック | 候補 | control | 改善量 |
| --- | ---: | ---: | ---: |
| Overall RMSE | `12.710664` | `11.938287` | `-0.772377 ft` |
| Raw GR observed | `12.526351` | `11.933740` | `-0.592611 ft` |
| Raw GR missing | `13.098358` | `11.948064` | `-1.150295 ft` |
| High missing wells | `13.352390` | `11.792411` | `-1.559979 ft` |
| MD 1000+ | `13.998399` | `13.135431` | `-0.862967 ft` |
| Hidden-like spatial | `14.618436` | `12.564491` | `-2.053945 ft` |
| Hidden-like typewell-purged | `14.476913` | `12.367244` | `-2.109669 ft` |
| Fixed LikPF 50:50 | `10.653104` | `10.269693` | `-0.383411 ft` |

- 改善fold: `1 / 5`（fold 3のみ`+0.327339 ft`）
- 改善well / 悪化well: `330 / 443`
- by-well RMSE差 p95: `+7.038260 ft`（上限`0`、FAIL）
- worst well: `e03b45fd`、`+46.046495 ft`（上限`+0.25`、FAIL）
- 固定した全scientific gateをAND判定し、FAIL。
- Public / Private LB: なし

## 技術監査

予測は`3,783,989 rows / 773 wells / 773 HMM runs`、finite coverage `1.0`、
posterior normalization最大誤差`4.21885e-15`、truth-before-freeze `0`で完走した。
保存された倍率は全773行で`1.3`、実効sigmaは`14.551610--78.0`。

実行済み`promotion_gate.json`の
`global_sigma_multiplier_contract_passed=false`は、in-memory runtime値と別CSVから
再読込したaudit値を`atol=0`で比較した監査上の偽陰性だった。独立再計算で差は最大
`2.1316282072803006e-14`に限られ、HMM計算への`1.3`倍適用は成立している。
ローカルsourceはCSV round-trip許容差`1e-12`へ修正したが、科学結果が大幅悪化しているため
再実行しない。

## 再現性

- Kaggle kernel:
  `kentookumura/exp398-all-well-sigma1p3-exact-hmm-train` version 1 /
  id_no `128542706`
- runtime: `19324.104 sec`（約5時間22分）、CPU、GPU/internet off
- executed source SHA:
  `1a7555296cd1f1e6eab354ed4afa728299d5b23fdd66aec37caf83bf9fc8e0b2`
- executed config SHA:
  `bec1374cfd2056433af1ed2bee01c5ce2adf148dbd98a32c94fc66d26e63257f`
- scientific contract SHA:
  `de38dbeaa3124522b2e3be9e22ae1cf7ea1a51cf157852c624b7a4e0a0ba08c6`
- prediction raw / content SHA:
  `288c1db45873d469ac516defb8f62e2df185ee581023317637a6f1675869a6f4` /
  `937c969e2a240fb02533bcb9b00cec31ce0dd0210e547a37c7df54acbd3f0b23`
- promotion gate SHA:
  `4b165b4539a1ef05f0d22b924b481a8e8b2e7508a969a89df5948efbc8741cc6`
- model / submission SHA: model・submissionなし

## 解釈

全well一律`1.3`倍は、GRが有効なwellまで一様に弱め、全体、4/5 folds、欠損・long-tail・
hidden-like、fixed blend、by-well tailのすべてで一貫して悪化した。selectorなしの固定global
interventionとしては不採用とする。

## 次

倍率、clip、emission、transition、grid、blendを同じOOFで調整せずbranchを閉じる。
exp397とexp398の結果から、GR sigma multiplier familyの追加救済backlogは作らない。
