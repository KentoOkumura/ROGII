# exp419_exp226_guided_defensive_mixture_pf 結果

## 状態

Kaggle train-side実行完了。technical gateはPASSしたが、mechanism gateと
standalone adoption gateはFAILした。事前登録どおり
`proposal_rejected_close_without_same_oof_rescue`でterminal closeする。

## 仮説

exp226のfold-safe geometry rateをimportance-corrected defensive proposalとして使うと、
PFのtarget posteriorを変更せずにfinite particle supportを改善できる。

## 固定設定

- Route: `pf_beam`
- control: 保存済みexp404 `likpf_scale_5_x1p0`
- candidate: exp226-guided proposal 1 variant
- proposal: original 0.5 + geometry `1x / 4x / 16x`各`1/6`
- correction: `p0/q`、clipなし、構成上上限2
- PF: 773 wells ×128 seeds ×500 particles、Kaggle CPU 4 shards
- control rerun / HMM / Beam / model / booster / GPU: 0
- merge kernel: version 1、id_no `128974840`

## 結果

| 指標 | candidate | 比較対象 | 差 / 判定 |
| --- | ---: | ---: | ---: |
| pooled RMSE | 10.680074 | exp404 10.914522 | +0.234448 ft改善 |
| 改善fold | 4 / 5 | 必要4 / 5 | PASS |
| raw-GR observed gain | 0.165544 ft | 必要0.10 ft | PASS |
| persistent episode SSE削減 | 14.8213% | 必要10% | PASS |
| support外率 | 97.4973% | exp410 64.2061% | 33.2912 pt悪化 |
| hidden-like spatial gain | -0.115823 ft | 悪化上限0.02 ft | FAIL |
| by-well delta RMSE p95 | +5.766213 ft | 上限+0.25 ft | FAIL |
| worst-well regression | +20.570238 ft | 上限+2.0 ft | FAIL |
| exp226 final比 | -1.252965 ft | 必要+0.03 ft | FAIL |
| exp226比改善fold | 1 / 5 | 必要3 / 5 | FAIL |

fold 0--3はexp404比で改善したが、fold 4は`0.151558 ft`悪化した。
well単位では463 wells改善、310 wells悪化し、最大悪化wellは`8902c3f6`だった。

## Technical gate

technical gateはPASSした。3,783,989 rows / 773 wellsを欠落・重複・fallbackなしで
mergeし、candidate coverageは1.0だった。freeze前のtruth / fold / hidden-like /
control / exp226 final / exp410 scope readはすべて0、proposal allowlistは
`well_id / row_idx / suffix_offset / tvt_geop`のみである。

importance ratio最大は`1.999999999999981`、geometry-weight-zero parity差は`0.0 ft`。
preflightとfull shardの固定well predictionはfloat32 byte-identicalだった。

## 解釈

importance correctionはtarget posteriorの契約を保ったが、有限500粒子の50%を
geometry成分へ割り当てることで、有限粒子supportは改善せず大幅に悪化した。
pooled RMSEと固定episode SSEの改善だけではこのsupport崩壊とwell-tail悪化を相殺できない。
exp226 geometry rateは一部区間の方向づけには効くものの、global defensive proposalとして
安全なsupport供給源ではない。exp226 final OOFにも大幅に劣るため、standalone PF候補、
deterministic anchor、inference / submission候補には昇格させない。

## 再現性と生成物

- prediction logical SHA:
  `2465d2aae907af57ad16daa0588d3210b8d201bad5f622f4408f5f4d3b701740`
- prediction raw gzip SHA:
  `0165104cc606c1a5d64f7682f9ae1ad946f8b5e490c0efc15bf8a10c06887789`
- prediction decompressed SHA:
  `8b7e8fc05a4cf529d0f9d4fe1cab8fc041c30eeb88e555e55f0aaf2c4255ea43`
- artifact manifest SHA:
  `fa7a2be4d494edf9813c56647fc52f8fba02f7697bcf3e2c14d1967bac66ce0b`
- scientific contract SHA:
  `a25d809a7af142b74f3d7e5a8eec7f54247aa1bbf659b2a646493277fc50f013`
- ローカル取得先:
  `kaggle/output/merge_v1/`

## 次

exp419内ではmixture weight、proposal幅、importance clip、GR sigma、process noise、
roughening、seed / particle数、well / row gateを救済探索しない。推論・提出も行わない。
保存済み生成物だけを使う低優先のsupport失敗原因readoutを候補化し、proposal型の
exp432はexp419のnegative evidenceを反映して優先度を下げる。
