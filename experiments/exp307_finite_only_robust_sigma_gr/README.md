# exp307_finite_only_robust_sigma_gr

## 状態

- Route: `pf_beam`
- 状態: Kaggle CPU train v2完了、全promotion gate FAILで閉鎖
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV / LB / submission: train-side paired RMSE確定 / なし / なし

## 仮説

既知prefix欠損GRの0補完が`σ_GR`を欠損率依存にしている。有限pairだけのMAD scaleならGR尤度のwell別校正とexact-HMMを改善できる。

## 固定した変更

- diagnostic: finite-only population std
- primary: `1.4826 * MAD`
- 20 pair未満は30、clip `[10,60]`
- evaluation GR補間、typewell、HMM grid/transition/posterior meanはexp209固定
- 2 variants、1,546 HMM well-runs、0 booster、control再実行0

## 検証方針

saved exact-HMMとのpaired train-side RMSEをoverall、5 folds、1000+、hidden-like 2面、by-well p95/worstで比較し、primary MADだけをpromotion候補とした。scale/predictionをcontent SHAでfreezeした後にtruthを結合し、fixed LikPF 50:50のnon-regressionも必須とした。

## 結果

finite std directは`11.938287 → 14.209718`、finite MAD primaryは`11.938287 → 15.661341`へ悪化した。fixed LikPF 50:50もそれぞれ`10.269693 → 10.767490 / 11.187333`へ悪化した。primaryはdirect 0/5 folds、blend 1/5 folds改善で、1000+、hidden-like 2面、p95、worstも全FAILした。

0補完除去によりscale中央値は旧`38.6418`からfinite std `13.8957`、finite MAD `10.1367`へ急低下し、MADは365/773 wellsで下限10に張り付いた。GR emissionの過信が増えたと解釈し、救済grid、inference、submissionを行わず閉じる。

Kaggle version 1はlate readout列契約で失敗したが、version 2はHMM前schema guardを追加して全計算を完走した。詳細とSHAは`SESSION_NOTES.md`と`result.md`を正とする。

## 所見

0補完が旧scaleを膨らませていたこと自体は確認できたが、その膨らみは現行Gaussian GR emissionの過信を抑える実効的な温度として働いていた。finite-only化を単独で採用せず、同一結果上のclip/sigma/likelihood救済も行わない。

## 次

exp307と固定dependency descendantsを閉じ、inference/submissionへ進まない。別parentで再設計する場合は新しい独立根拠と事前設計、ユーザー確認を必要とする。
