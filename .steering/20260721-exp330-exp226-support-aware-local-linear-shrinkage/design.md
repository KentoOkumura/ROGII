# 設計

## アプローチ

exp226のlocal-linearはtarget segmentごとにk50 donorへXY局所平面をfitし、そのinterceptをdriftとして使う。donor supportが弱いsegmentでは、同じdonorとweightから傾きを持たないweighted constantを計算し、exp329 risk q80以上だけ平面interceptを最大50%その値へ戻す。

```text
activation = clip((risk - 0.80) / 0.20, 0, 1)
alpha      = 0.50 * activation
h_reg      = h_linear + alpha * (h_constant - h_linear)
```

raw/smoothed fieldへ同一式を適用する。donor identity、weight、distance、bucketは変えない。保存済みfold kappaを再fitせず使うため、介入はtarget-wellの空間外挿だけに限定される。

## 実験範囲

- 対象: `exp330_exp226_support_aware_local_linear_shrinkage`
- Route: `pf_beam`
- 親: exp226保存OOFとfold別kappa。
- 変更: high-risk K16 segmentのraw/smoothed local-linear intercept。
- 固定: kappa、K16、k50、bandwidth、ridge、distance bucket、ANCC、GR、U projection。

## 段階

Stage 0は固定32 wellsで`risk=0` parent parity、kappa SHA、exp329 contract parity、identity、finite、runtimeだけを監査する。Stage 1はreal配置1本と同数circular control 1本のfull OOF。各段階は別承認とし、FAIL時はparameter救済をしない。

## 再現性設計

RNGなし。exp226 source fold、well、segment、donor順を固定する。parent/constant/regularized field、risk、kappa、predictionをcontent hashする。control offsetだけSHA256(well id)で固定する。

## リスク

- weighted constantが平面よりbiasを持つ: top-riskだけ・最大50%に制限し、circular control差とtop-risk gainを要求する。
- kappaとの不整合: これは意図的な単一介入だが、親kappaを固定して原因分離し、overall/fold/tail hard guardで判定する。
- GR/U下流差の混入: algorithm/parameterは固定し、regularized fieldを既存経路へ渡す必然的下流結果として事前登録する。
- exp329 riskが不支持: Stage 0 FAILなら本実験を開始しない。
