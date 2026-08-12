# exp376_exp226_formation_conditioned_k16_donor_kernel

## 状態

- ルート: `pf_beam`
- 状態: `kaggle_cpu_v2_completed_technical_pass_direct_fail_novelty_fail`
- 作成日: 2026-07-24
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- CV / LB / Submit ID: 未実行のためなし

## 仮説

exp226の正解TVT由来K=16 slopeはそのまま使い、同じXY近傍50 donorへのweightを
fold-safeな6地層の相対座標で緩やかに条件付けすれば、空間的に近いが異なる地層位置の
donorを弱めつつ、既存のsupportを失わずにTVT drift補間を改善できる。

## 単一変更

segment midpointで推定した6地層面から、面相対距離6個と隣接面厚5個を作る。
outer-train donorだけでrobust標準化し、次の固定式を既存XY weightへ掛ける。

```text
g_form = 0.5 + 0.5 * exp(-0.5 * d_form^2)
w_new  = w_xy * g_form
```

nonfinite時は`g_form=1.0`へ戻す。K16、raw/smoothed slope、近傍50、
bandwidth、ridge、rho、adaptive kappa、ANCC local theta、GR correction、
U-projectionは親のまま固定する。

## 検証方針

- outer 5-fold / well group。outer-valid wellのTVTと6地層列をreferenceへ使わない。
- outer-train donorの地層signatureもself-exclusionで推定する。
- Stage 0でleakage、finite、fallback、support、SHAをtruth-freeに監査する。
- PASS時だけ保存済みexp226 OOFとのdirect比較と、exp293 fixed12への
  H512/whole-well add-one noveltyを評価する。
- 予定量は1 variant / 0 model config / 5 reporting folds /
  0 trained fold / 0 booster / parent control再実行0。

詳細と事前gateは
`docs/legacy/steering/20260724-exp376-exp226-formation-conditioned-k16-donor-kernel/design.md`
を正とする。

## 実装状態

`exp376_exp226_formation_conditioned_k16_donor_kernel_compact_selfcontained_train.py`
と対応する`.ipynb`を実装した。9章・19セルで、次をnotebook内に展開している。

- exp226のK16 donor slope、adaptive kappa、ANCC local theta、GR correction、
  U-projectionを含むtrain-side full downstream。
- exp287と同じwell-median / k=10 / self-exclusion semanticsの
  fold-local `FormationPlaneKNN`。
- outer-train donorだけで作る11次元signature、median/MAD標準化、
  同じXY近傍50への固定soft factor。
- truthを開く前のreference/signature/support/prediction/SHA freezeと
  Stage 0 fail-closed gate。
- Stage 0 PASS時だけ実行するexp226 direct比較とexp293 fixed12への
  H512/whole-well add-one novelty。

ユーザーの実行指示を受け、compact版を正規`*_train.ipynb`へ採用し、
Kaggle CPU runを1回実行する。正規`*_inference.ipynb`はtemplate scaffoldの
まま変更しない。current-test、推論、提出は未承認。

## 結果

Kaggle CPU v1は5 foldsの予測後、truth前freezeでreference manifest内のlist列を
pandas hashへ渡して`TypeError: unhashable type: 'list'`となった。
Stage 0/1/2とCVは未評価で、truthは開いていない。

container-valued cellだけをcanonical JSONへ正規化してからlogical hashする
局所修正を正規notebookへ反映し、Kaggle CPU v2を完走した。

- Technical / Stage 0: PASS / PASS
- Direct: `9.443257190`、exp226 `9.427109597`比`+0.016147593 ft`でFAIL
- by-well p95 / worst: `+0.376679 / +1.891560 ft`でFAIL
- H512 / whole-well add-one改善:
  `0.019403532 / 0.015542019 ft`で閾値`0.05 ft`未達
- Candidate novelty: strict unique-best `9.387441%`、5/5 foldsはPASSしたが総合FAIL
- Decision: `close_formation_conditioned_donor_branch_without_rescue_grid`

専用test 4件、`py_compile`、ruff F821、Jupytext round-tripもPASSした。

## 所見

exp287からformation signalの可能性はあるが、tail悪化も確認されている。
そのためpooled RMSEだけでは採用せず、target-free support guardとwell-tailを
hard gateに含めた。v2ではsupportを安全に維持したが、directとtailを悪化させ、
fixed12への増分も小さかったため不採用とする。救済grid、推論、提出は行わない。
