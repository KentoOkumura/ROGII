# exp330_exp226_support_aware_local_linear_shrinkage

## 状態

- Route: `pf_beam`
- 状態: 必須依存FAIL・未実装/未実行のままbranch closed
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 必須依存: exp329のdonor-support risk readout全gate PASSとcontract SHA固定。exp329 Stage 1の成否は依存しない。

## 仮説

exp226の外れがdonor不足やlocal-linear extrapolationの不安定性から生じるなら、高risk K16 segmentだけ局所平面のinterceptを同じdonorの重み付き定数推定へ戻すと、空間勾配の過剰外挿を抑えながらdonorの平均dip情報は保持できる。

raw/smoothed fieldの両方に同じ式を使う。

```text
g_j       = clip((risk_j - 0.80) / 0.20, 0, 1)
a_j       = 0.50 × g_j
h_const,j = sum(w_i × donor_drift_i) / sum(w_i)
h_reg,j   = h_linear,j + a_j × (h_const,j - h_linear,j)
```

`h_const`は親local-linearと完全に同じk50 donor、bandwidth 500のweightを使う。最大でも平面推定から定数推定へ半分しか移さない。zero driftやZ-onlyへは戻さない。

## 変更しないもの

- K=16、k=50、bandwidth 500、ridge 1、donor距離とbucket。
- 保存済みfold別kappa。kappaは再fitしない。
- near-strike ANCC committee、GR correction、U projection。
- exp226のfoldとvalidation-fold donor除外。
- HMM/PF/Beam/model。特にexp324のsegment別`sig_r,t`とは別介入である。

regularized raw fieldを既存relpath/GR correctionへ渡し、raw/smoothed fieldを既存design matrixへ渡すことは1つのfield変更の下流結果として固定し、別parameterにはしない。

## 検証方針

### Stage 0: parity preflight

exp329 Stage 0 PASS後、実装承認があった場合だけ各fold均衡の固定32 wellsで次を確認する。

- `risk=0`で保存済みexp226 predictionとの差が最大`1e-8 ft`以下。
- fold別kappa file SHAが`6cbded4...d1aeff0`と一致し、係数差0。
- exp329 support primitive/risk contract、donor id/weight/distance/segmentが一致。
- finite coverageと8.5時間以内のruntime外挿。

1つでもFAILならfull OOFへ進まない。

### Stage 1: fixed one-variant OOF

Stage 0 PASSと別承認後だけ、real risk配置1 variantとwithin-well circular risk control 1本を評価する。保存済みexp226 OOFをcontrolとし、親のfull再実行やkappa fitはしない。

## 判定

- exp226比RMSEを`0.05 ft`以上改善し、4/5 folds改善。
- top-risk decileを`0.10 ft`以上改善。
- 0--250 ft、1000+、hidden-like 2面、by-well p95は非悪化。
- worst wellは`+0.25 ft`以内。
- real配置のgainがcircular controlを`0.02 ft`以上上回る。

## 実行量と境界

Stage 0は32 parity wells。Stage 1最大は1 scientific variant + 1 deterministic control、5 field-pack builds、合計1,546 target-well prediction runs、kappa fit/model/booster/HMM/PF/Beam 0。実装、実行、inference、submissionはまだ行わない。

## 所見

exp329 Stage 0はtechnical/coverage checksを全PASSしたが、pooled AUC 0.562091、control差0.005310、top-risk benefit -0.674259 ftでscientific gateをFAILした。したがって本実験が必要とするsupport-risk contractの採用条件は成立しない。

## 次

dependency policyどおり、Stage 0 parity、full OOF、inference、submissionを実装・実行せず閉じる。同じriskを使う救済gridは追加しない。
