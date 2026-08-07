# exp424_exp209_momentum1_exact_hmm_ablation 結果

## 状態

Kaggle private CPU Version 1でStage 0を完走した。technical gateは全PASSしたが、
mechanism gateは3 / 7 PASSで`stage0_fail_closed`。Stage 1、inference、
submissionは実行しない。

## 仮説

exp209 exact HMMのrate momentumを`0.998`から`1.0`へ変え、
0方向mean reversionだけを除けば、`sig_r=0.002`を増やさずにrate under-responseと
persistent TVT offsetを減らせる可能性がある。

## 実行

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 唯一の変更: `mom=0.998 -> 1.0`
- 固定: `sig_r=0.002`とその他すべてのHMM grammar / readout
- canonical kernel:
  `kentookumura/exp424-exp209-momentum1-exact-hmm-ablation-train`
- kernel version / id: `1 / 128924158`
- Stage 0: baseline 32 + treatment 32 = 64 HMM runs
- runtime / peak RSS: `2,077.533832秒 / 1.030926 GB`
- full 773-well treatment投影: `24,402.685805秒`
- model / booster / PF / Beam / GPU: 0

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0判定 | **FAIL_CLOSE_BRANCH** |
| technical gate | 13 / 13 PASS |
| mechanism gate | 3 / 7 PASS |
| persistent episode SSE削減 | `0.475550%`（閾値`>=5%`、FAIL） |
| persistent改善well | `8 / 16`（閾値`>=10`、FAIL） |
| persistent改善fold | `3 / 5`（閾値`>=4`、FAIL） |
| under-response SSE share低下 | `9.849995 points`（閾値`>=2`、PASS） |
| matched control pooled delta | `-0.054769 ft`（上限`+0.02`、PASS） |
| matched control by-well p95 | `+0.157066 ft`（上限`+0.25`、PASS） |
| smoothed rate edge mass delta | `+0.000377954`（nonworse、FAIL） |
| CV | なし（fixed32はmechanism-only） |
| Public LB | 未提出 |
| Private LB | 未提出 |

technical gate:

- baselineと保存済みexp209の最大差: `0.0 ft`
- posterior normalization最大誤差: `2.18467e-08`
- finite coverage: `1.0`
- nonfinite rate moment: `0`
- truth / episode read-before-freeze: `0 / 0`
- prediction / rate readoutのreadback logical SHA: 一致
- runtime projection / peak RSS: PASS

fold別persistent episode SSEはfold 0 / 1が悪化し、fold 2 / 3 / 4だけが改善した。

## 解釈

`mom=1.0`はrate under-responseのSSE占有率を大きく下げ、matched controlも安全側だった。
しかし主目的のpersistent TVT episode SSE改善は`0.48%`に留まり、well・foldの
一貫性も不足した。さらにrate-grid edge massが増えたため、0方向収縮の除去だけでは
persistent offsetを安定して修復できない。

fixed32はerror-selected mechanism sampleでありCVではないが、その有利な診断sample上でも
事前固定gateを満たさなかった。設計どおり`mom`、`sig_r`、sample、gate、blendを
same-OOFで救済せず、このbranchを閉じる。

## 次

Stage 1、inference、submissionへ進まない。momentum単独familyは再実行せず、
別仮説が必要な場合は独立したsteeringと実験番号で事前登録する。
