# exp495 exp226 rate不確実性重み付き観測HMM 結果

## 状態

Stage 0Aは事前gateをFAILした。ユーザーの明示overrideで、事前登録済みの条件を
変更せずStage 0B fixed32をKaggle private CPU version 4で実行したが、technical
1件とmechanism全7件をFAILしたため、Stage 1へ進まずfail-closedとした。

## 仮説

exp355で確認したexp226 geometry相対rateの平均signalを、known-prefix rate残差から
推定した不確実性付き観測としてexp209 rate transitionへ融合すれば、exp355の
worst-well悪化を抑えながらrate追従遅れを改善できる。

## 固定設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- fixed32: persistent 16 / matched control 16、156,088 suffix rows、5 folds
- rate center: exp355互換のexp226 geometry相対U-rate
- uncertainty: known-prefix tail128 centered MAD、floor `0.002`
- HMM: absolute rate 41 states、span `0.10`、`sig_r=0.002`、momentum `0.998`
- rate factor: Gaussian、追加scale / temperature / clip / gateなし
- TVT遷移: `ΔTVT = r_j × ΔMD − ΔZ`
- 保存済みexp209 / exp355を比較に使い、親/control HMM再実行は0

## 実行量

| 項目 | 数 |
| --- | ---: |
| scientific variant | 1 |
| candidate HMM well-run | 32 |
| parent/control HMM再実行 | 0 |
| model / booster / PF / Beam / GPU | 0 / 0 / 0 / 0 / 0 |

fixed32はmechanism preflightであり、CVやpromotion evidenceではない。

## Stage 0B RMSE結果

| 比較 | candidate | 保存baseline | candidate差 | 基準 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| all 32 vs exp355 | 13.069257 | 10.677951 | +2.391305 ft | 0.10 ft以上改善 | FAIL |
| persistent 16 vs exp355 | 16.344367 | 14.200976 | +2.143391 ft | 0.10 ft以上改善 | FAIL |
| matched control 16 vs exp209 | 8.454838 | 3.428436 | +5.026402 ft | +0.02 ft以下 | FAIL |

fold別ではexp355比でfold 1が`-1.470912 ft`、fold 3が`-0.524683 ft`と改善したが、
改善は2/5 foldsで基準4/5に届かなかった。fold 0は`+10.318903 ft`悪化した。

その他のmechanism gateも全てFAILした。

- persistent episode SSE reduction: `0.069667`（基準`0.10`以上）
- paired by-well delta p95 vs exp355: `+16.564282 ft`（基準`+0.25 ft`以下）
- worst-well delta vs exp355: `+23.911032 ft`（基準`+2.0 ft`以下）

平均でもwell-tailでもGaussian rate観測が強すぎる悪化を生み、known-prefix MADによる
縮約では危険wellを十分に保護できなかった。

## Technical gate

fixed32の32 wells / 156,088 rows / 5 folds、finite coverage 1.0、truth / role /
episode pre-freeze read 0、uniform-factor parent parity `4.58e-17 ft`、transition row-sum
error `1.40e-14`、projected full runtime `18,948 sec`、peak RSS `1.305 GB`はPASSした。

posterior normalization max errorだけが`5.6292e-06`で、基準`1e-06`をFAILした。
mechanism gateも全FAILのため、この数値許容差を変更して再実行しない。

## 再現性

- kernel: version 4 / id_no `129285050` / runtime `923.702763 sec`
- candidate HMM time: `784.397362 sec`
- input manifest SHA: `22d6ca0764661b4d8faeab03a7827ce060131f546ec7290cffff0462ca6f25f4`
- scientific contract SHA: `2760e9a5b96a5bd8eacd0ca329ebceac0c89555e74bf3cc7e7cb3180f4d98313`
- prefix uncertainty SHA: `2e36d48081d5bd851d6a8e49a777c291af124a5dddedb5830fc1d4fb9fb46a37`
- rate schedule SHA: `e31f048aeb8356a894d794d6c0cb1730006ebbe34c8282074a26ae0c445e8b59`
- prediction SHA: `e550b0cc7fadfad38a4f0606f36eacdb3dc189c63029adcc355dde59bc17e84e`
- deterministic anchor: false
- inference / submission: なし

## 結論

decisionは
`close_without_sigma_window_scale_temperature_emission_grid_blend_selector_or_pf_rescue`。
Stage 1へ進めず、window、sigma floor/scale、temperature、gate、emission、grid、blend、
selector、PFで同一fixed32を救済しない。保存済みStage 0A/0B artifactを使う原因分解だけを
低優先度readout候補として残す。
