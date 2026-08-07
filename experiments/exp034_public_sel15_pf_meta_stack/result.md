# exp034_public_sel15_pf_meta_stack 結果

## 仮説

`exp026` の pseudo-tail bucket-shrink 基準 と公開 sel15 PF/Beam prediction は誤差構造が違うため、fold-safe な 2nd stage で clean train well の途中以降を隠した疑似 test CV を改善できる可能性がある。

## 設定

- 親: `exp029_public_sel15_pf_oof_feature_generation`
- self-route 基準: `exp026_pseudo_tail_bucket_shrink_inference_submit` clean CV 12.870780
- 検証: original-fold OOF、stable well-hash holdout
- メトリック: RMSE
- シード: 42
- Required controls: `exp026_pseudo_tail_bucket_shrink`、`public_pf_selector`、`pf090_hold010`
- Kaggle kernel: `kentookumura/exp034-sel15-pf-meta-stack-train` version 2

## 結果

| メトリック | 値 |
| --- | --- |
| rows / wells | 1,782,279 / 773 |
| Required control, original-fold | 15.089532 |
| Required control, well-hash | 15.089532 |
| Best original-fold OOF | 14.313668 (`ridge_meta_residual_shrink0p75_clip60p0`) |
| Best well-hash holdout | 14.172010 (`ridge_meta_residual_shrink0p75_clip60p0`) |
| Public LB | - |

Selected `ridge_meta_residual_shrink0p75_clip60p0` は original-fold で exp029-row `exp026_pseudo_tail_bucket_shrink` control から -1.330065、public PF から -0.858969、`pf090_hold010` から -0.775864 改善した。well-hash では exp026-row control から -1.654092、public PF から -1.000626、`pf090_hold010` から -0.917521 改善した。

距離 bucket 別にも selected は両 audit の全 bucket で exp026-row reference を上回った。original-fold の改善は rows 0-49: -1.169745、50-249: -1.214705、250-999: -1.256925、1000-2499: -1.371501、2500+: -1.642534。well-hash では rows 0-49: -1.507170、50-249: -1.530323、250-999: -1.563184、1000-2499: -1.678403、2500+: -2.260803。

Split 別でも selected は全 split で exp026-row reference より改善した。original-fold split delta は -0.258764、-1.856668、-1.299029、-1.511069、-1.726921。well-hash split delta は -1.529526、-2.197444、-2.213784、-1.351723、-1.082954。

## 解釈

public sel15 PF/Beam artifact と exp026-style 基準 の間には、ridge で拾える強い linear stacking signal がある。HGB residual は supported candidate には残ったが、ridge 上位より弱く、まずは selected ridge だけを推論移植候補にする。

ただし、この評価は exp029 train well の途中以降を隠した疑似 test row の評価条件 上の audit であり、exp026 clean CV 12.870780 と同一 surface ではない。exp031 / exp033 では train-side 改善が 見えない test well 評価の LB に転移しなかったため、今回の supported candidate も即 submit せず、別実験で inference port、output diff、range、start continuity、Public LB 基準 8.781 との関係を監査する。

## 次

`ridge_meta_residual_shrink0p75_clip60p0` を exp027 public sel15 inference flow に移植する別実験を作る。提出前に exp027 output との差分、見えない test well 用処理の発火範囲、予測範囲、submit-check を確認する。
