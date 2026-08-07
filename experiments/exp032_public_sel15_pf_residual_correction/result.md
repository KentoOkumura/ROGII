# exp032_public_sel15_pf_residual_補正 結果

## 仮説

`exp029` の public sel15 PF/Beam OOF-like artifact には、`pf_pred` の系統誤差を補正できる 見えない test で使える signal が残っている可能性がある。

## 設定

- 親: `exp029_public_sel15_pf_oof_feature_generation`
- 検証: original-fold OOF residual prediction、stable well-hash holdout residual prediction
- メトリック: RMSE
- シード: 42
- Required controls: `public_pf_selector`、`pf090_hold010`
- Kaggle kernel: `kentookumura/exp032-sel15-pf-residual-train` version 2

## 結果

| メトリック | 値 |
| --- | --- |
| Public PF selector | 15.172636 |
| PF 90 / hold 10 control | 15.089532 |
| Best original-fold OOF | 14.937393 (`ridge_residual_shrink0p5_clip20p0`) |
| Best well-hash holdout | 14.844228 (`ridge_residual_shrink0p5_clip20p0`) |
| rows / wells | 1,782,279 / 773 |
| Public LB | - |

Selected `ridge_residual_shrink0p5_clip20p0` は original-fold で PF 単体から -0.235243、`pf090_hold010` から -0.152138 改善した。well-hash holdout では PF 単体から -0.328408、`pf090_hold010` から -0.245304 改善した。

距離 bucket 別にも selected は original-fold / well-hash の両方で PF 単体を上回った。original-fold の改善は rows 0-49: -0.190268、50-249: -0.160953、250-999: -0.120403、1000-2499: -0.296725、2500+: -0.375979。

## 解釈

PF の残差には linear ridge で拾える安定した補正信号がある。HGB residual は original-fold で `pf090_hold010` とほぼ同等か悪化し、強い非線形補正は過補正しやすい。clip 20 / shrink 0.5 が最良で、残差補正は保守的に入れる方が良い。

一方、original-fold split 3 だけは selected が PF 単体より +0.074417 悪化した。全体と bucket は改善しているが、推論化する場合は 見えない test well 用処理の output diff、range、start continuity、Public LB 基準 8.781 への影響を別実験で監査する。

## 次

`ridge_residual_shrink0p5_clip20p0` だけを exp027 public sel15 inference flow に移植する別実験を作る。提出前に exp027 / exp031 との差分と submit-check を確認する。
