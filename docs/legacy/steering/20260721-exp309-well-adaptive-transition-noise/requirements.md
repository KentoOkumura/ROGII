# 要件

## 依頼

- `well_adaptive_transition_noise`を`exp309`として新規作成し、設計を確定する。
- 既知prefixの`U=TVT+Z` rate変動からexact-HMMのrate diffusion `sig_r`をwell別に推定する。
- 今回はdesign-onlyとし、実装、Kaggle実行、inference、submissionは行わない。

## 2026-07-21 追加承認

- ユーザーの「exp309を実装してください」「実装だけ先に進める」を実装承認として追加した。
- exp308未完了でもコード、Notebook、contract test、静的検証は先行する。
- exp308のpromotion status、prediction SHA、親metricsが未確定の間は実行入口をfail-closedにする。
- Kaggle package、push、Notebook実行、inference、submissionは引き続き承認対象外とする。

## 仮説

全well共通`sig_r=0.002`は、滑らかなwellで探索を広げすぎ、rate変化の大きいwellで狭すぎる。prefixから推定したrobust rate-increment scaleをglobal値へ縮約すれば、地層変化のwell差を状態遷移へ反映できる。

## 制約

- Routeは`pf_beam`、親は`exp308_imputed_gr_confidence_downweight`とする。
- exp308が全promotion gateをPASSし、input/prediction SHAが固定された場合だけeligibleとする。
- 変更は`sig_r`だけ。`sig_p=0.02`、position kernel floor、momentum、rate grid/range、GR emission/scale/weight、posterior meanを固定する。
- known prefixの`q_i=Δ(TVT_input+Z)/ΔMD`と隣接差からrobust diffusion scaleを作る。
- valid rate-increment 20未満はglobal `0.002`。scaleをsupport shrinkageし、`[0.001,0.004]`にclipする。
- clip、shrinkage strength、tail length、`sig_p`、momentum、rate state gridを探索しない。
- 1 variant x 773 = 773 HMM well-runs、model/LightGBM/fold/booster/PF/Beamは0、parent再実行0とする。

## 受け入れ基準

- exp308 dependency/SHA、rate formula、finite/support/fallback/clip、truth late-joinをPASSする。
- adaptive `sig_r`がparentよりoverall RMSEを0.05 ft以上、4/5 foldsで改善する。
- 1000+、hidden-like 2面、by-well p95を悪化させず、worst regressionを+0.25 ft以下にする。
- `sig_r` fallback率50%未満、上下clip合計率50%未満とし、適応が実質固定/clip ruleになっていない。
- fixed LikPF 50:50をparent blendから悪化させない。
- FAIL後のsig_r range/shrinkage/momentum/sig_p/rate-grid救済へ進まない。

## 次のアクション

実装後はexp308 PASSとdependency SHA/metrics固定までKaggle実行をブロックする。
