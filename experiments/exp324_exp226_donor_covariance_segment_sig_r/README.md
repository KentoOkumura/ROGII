# exp324_exp226_donor_covariance_segment_sig_r

## 状態

- Route: `pf_beam`
- 状態: 閉鎖済み・未実装・未実行
- 親: `exp323_time_varying_exp226_dip_rate_prior`

親exp323がterminal closeしたため、2026-07-22に本実験も閉鎖した。新exp324相当は、exp338 PASS後に作る新exp323相当がさらにPASSした場合だけ、新番号で設計する。

## 仮説

exp226が各K16 segmentで参照する近傍donorのrate分散をHMMの`sig_r,t`へ変換すると、donorが一致する区間では余計なrate移動を抑え、不一致区間ではexp323 priorから安全に外れられる。

```text
s_geo,j = 1.4826 × weighted MAD(projected donor rates)
alpha_j = n_eff / (n_eff + 50)
log(sig_r,j) = (1-alpha_j)log(sig_r,parent) + alpha_j log(s_geo,j)
sig_r,j = clip(sig_r,j, 0.001, 0.004)
```

donor有効数10未満は親well別`sig_r`へfallbackする。区間内は一定とし、rate prior平均はexp323から一切変えない。

## 段階

- Stage 0: sigma scheduleをtruth-freeに凍結し、transition NLLと68/95% calibrationを定数sigmaと比較する。HMM 0。
- Stage 1: Stage 0全PASSと別承認後だけ1 variant / 773 HMM runsを許可する。

## 検証方針

- Stage 0: 定数sigma比でtransition NLLを1%以上改善し、4/5 folds、68/95% calibration、1000+・hidden-like・worst、fallback/clip guardを全PASSする。
- Stage 1: 保存済みexp323親HMM比0.05 ft以上、4/5 folds、p95/worst `<=+0.25 ft`を要求する。
- donor identity、weight、`n_eff`、segment schedule、fallback/clipとcontent SHAをsuffix truthの結合前に固定する。

## 所見

prior平均を変えず分散だけを変えるため、exp323のgeometry meanと原因分離できる。低支持区間は親へfail closedする設計であり、現時点では結果はない。

本実験の実装、実行、inference、submissionは今後行わない。
