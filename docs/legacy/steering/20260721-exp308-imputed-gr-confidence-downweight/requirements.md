# 要件

## 依頼

- `imputed_gr_confidence_downweight`を`exp308`として新規作成し、設計を確定する。
- evaluation区間で補間されたGRを実測GRと同じ強さで数える現行exact-HMMを、元の欠損位置と実測GRまでの距離に応じたsoft confidenceへ変更する。
- 初回はdesign-onlyとし、その後のユーザー依頼「exp308を実装してください」を実装のみの承認として扱う。
- exp307 PASS前でもコード、Notebook、contract testまでは実装するが、dependency SHA/metricsは`PENDING`のままにし、Kaggle package/push/runはfail-closeする。
- inference、submissionは引き続き行わない。

## 仮説

短い欠損の線形補間には局所情報があるが、長い欠損中央の補間値を実観測と同じ尤度で繰り返すとwrong modeを過信する。補間値自体を変えず距離依存でGR emissionを弱めれば、exp269のblanket neutralityを避けながらtail errorを減らせる。

## 制約

- Routeは`pf_beam`、親は`exp307_finite_only_robust_sigma_gr`とする。
- exp307 primaryが全promotion gateをPASSし、prediction/scientific contract SHAが固定された場合だけ実装・実行eligibleとする。
- raw GRが有限な行はweight 1。元が欠損の行は最近傍raw finite GRまでの同一well行距離`d`を使い、`w=max(0.25, 2^(-d/8))`に固定する。
- exp307の線形補間値、typewell GR、finite MAD `σ_GR`、exact-HMM grid/transition/prior/posterior meanを固定する。
- weight/floor/half-life/run-length grid、blanket mask、0 weight、GR再補間、temperature救済を行わない。
- 1 variant x 773 = 773 HMM well-runs、LightGBM/model/fold/booster/PF/Beamは0、parent control再実行0とする。

## 受け入れ基準

- exp307 dependency/SHA、raw missing mask、補間GR parity、observed-row weight exact 1、missing-row weight `[0.25,1)`をPASSする。
- parent exp307 primaryよりoverall RMSEを0.05 ft以上、4/5 foldsで改善する。
- missing/observed row別、short gap 1--3、medium 4--15、long 16+のreadoutを保存する。
- 1000+、hidden-like 2面、by-well p95を悪化させず、worst regressionを+0.25 ft以下にする。
- saved LikPFとの固定50:50をparent blendから悪化させない。
- FAIL後のhalf-life/floor/grid、exp269 rescue、PF/LikPF port、inference、submissionへ進まない。

## 次のアクション

exp307 version 2がpromotion gateをFAILしたため、dependency SHA/metricsをexp308の実行入力として固定しない。実装と静的/synthetic検証は履歴として保持し、Kaggle package/push/run、inference、submissionなしで閉鎖する。
