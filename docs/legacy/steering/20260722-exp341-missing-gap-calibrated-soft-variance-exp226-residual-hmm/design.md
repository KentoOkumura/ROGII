# 設計

## アプローチ

exp339のouter-train tableを対応outer-valid wellへ適用し、exp281のrow-dependent GR emissionでmissing rowの分母だけを広げる。補間平均を変えず、state-neutral化もしない。

```text
observed row: sigma_eff_t^2 = sigma_GR^2
raw missing row: sigma_eff_t^2 = sigma_GR^2 + sigma_imp(L,d)^2
ell_t(delta) = -0.5 * min((GR_interp_t - GR_tw(TVT_geop_t + delta))^2 / sigma_eff_t^2, 600)
```

## 実験範囲

- 対象実験: `exp341_missing_gap_calibrated_soft_variance_exp226_residual_hmm`
- Route: `pf_beam`
- 親実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- 依存: exp339全gate PASS。
- 変更する変数: raw missing rowのGaussian varianceだけ。
- 固定する変数: GR補間値、observed row、base sigma、exp226 shape、offset HMM grammar、grid、transition、prior、output。
- 実行量: 1 variant、773 HMM well-runs、control再実行/model/booster 0。

## 検証方法

1. exp339 table/fold、exp281/exp226 control、raw mask SHAをHMM前にpreflightする。
2. missing confidence scheduleとscientific contract SHAをtruth join前にfreezeする。
3. 1 variantを生成後、prediction content SHAをfreezeする。
4. suffix truth、fold、distance、hidden-like、by-wellをlate joinしAND gateを判定する。

## 再現性設計

- RNGなし。fold/well/raw row/table lookup順固定。
- Kaggle CPU、GPU/internet off、最大1 variant・8.5時間。
- input/table/mask/schedule/prediction/metricsのdecompressed content SHAを記録する。
- model/submission SHAは非該当。inference/submissionはdisabled。

## リスク

- exp269のnegative: 補間evidenceを完全には外さず、exp339校正分だけsoft化する。
- closed branch rescue: exp339の独立pseudo-gap evidenceがPASSした場合だけ新番号で実施する。
- objective mismatch: interpolation NLL改善はTVT RMSEを保証しないためfull HMM gateを維持する。

## 優先度

exp339 PASS時のみ`P2`。未PASSのまま実装しない。
