# 設計

## アプローチ

outer foldごとにouter-train known prefixのraw missing-run histogramを作る。outer-valid wellのfinite known-prefix runから、同histogramを満たすinterior pseudo-gap候補をstable SHA256で選び、exp209互換linear interpolation後の誤差をlate joinする。uncertainty tableはouter-train pseudo-gapだけでfitし、outer-validへ固定適用する。

```text
e_imp = GR_hidden_raw - GR_linear_interpolated
v_cell = shrink(mean(e_imp^2 | L_bin, d_bin), length_bin, global; k=200)
sigma_imp = sqrt(v_cell)
```

primaryは2D table、controlはouter-train global constant variance、negative controlはwell内でpseudo-gap placementをstable circular rotationしたmatched controlとする。exact lengthはouter-train histogramのbin内CDFへ`fold|well|length_bin|slot`のstable SHAを当てて決める。coverageはouter-valid wellのうち1件以上pseudo-gapを作れたwellの割合とする。circular controlはreal planとgap identity・長さ・件数を一致させ、各gapのfinite-anchor候補列をstable non-zero offsetで循環して配置し直す。移動不能なgapだけ同一位置を許容し、その件数をauditへ残す。TVT predictionは作らない。

## 実験範囲

- 対象実験: `exp339_missing_gap_pseudomask_uncertainty_readout`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: なし。missing interpolation errorのtarget-free uncertainty readoutだけを作る。
- 固定する変数: raw row order、fold、exp209 interpolation、run/distance bin、support `k=200`、pseudo-gap上限4、controls、gate。
- 実行量: 1 scientific readout + 2 controls、HMM/model/config/booster各0。

## 検証方法

1. raw well identity、fold manifest、known-prefix/raw-missing maskをpreflightする。
2. outer-trainだけでrun histogram、pseudo-gap table、fallbackをfreezeする。
3. outer-valid pseudo-gap identityとinterpolation prediction SHAをfreeze後、隠したGRをjoinする。
4. pooled/fold/bin/coverage/NLL/calibration/control差をAND gateで判定する。
5. fold manifest、natural missing inventory/histogram、gap plan、hidden-GRなし補間予測、2D table、late-join audit、fold summaryをdecompressed content SHA付きで保存する。

## 再現性設計

- seed policy: RNGなし。SHA256キー`fold|well_id|length_bin|start_row`で候補順を固定する。
- stochastic処理、PF/Beam、GPU学習: なし。
- CPU、internet off。並列化しても候補identityと集約順を固定する。
- raw/fold/pseudo-gap/table/prediction/auditのschema SHAとdecompressed content SHAを記録する。
- model/submission SHAは非該当。deterministic submission anchorとは扱わない。
- package時はcanonical kernel metadataとbootstrap内config/source SHAを照合する。

## リスク

- leakage: outer-valid errorをtable fitへ入れない。unknown suffix TVTは全工程で不要。
- distribution shift: known-prefix pseudo-gapとsuffix natural gapが異なる可能性をmatched natural-run histogramとfold別calibrationで監査する。
- selection bias: placementやbinを結果後に変更しない。
- runtime: 全finite window列挙を避け、stable SHA上位4件へ制限する。

## 優先度

Late phaseの`P1` 0-HMM診断。高コストHMM案より先に安価に反証する。
