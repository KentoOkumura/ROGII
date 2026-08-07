# exp039_ravaghi_single_lgbm_inference_submit 結果

## 状態

Kaggle inference v2 完了、submit-check PASS、code competition submit 完了。Public LB は 11.740。

## 評価内容

exp038 selected `base_plus_pf_prediction_bucket_shrink` を見えない test well 用処理に移植し、public sample の visible 物理処理が unchanged であることを確認して code competition submit した。

## 結果

- inference: `kentookumura/exp039-ravaghi-lgbm-infer` version 2 completed.
- submit-check: PASS.
- submission ref: `53464736`.
- submission SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`.
- public sample diff vs exp027 基準: changed_rows=0, changed_wells=0, diff_rmse=0.0.
- Public LB: 11.740.

## 解釈

public sample output は exp027 基準と同一。これは visible public sample wells が physical/PF branch のままで、single-LGBM residual 補正が見えない test well 用処理専用のため。

見えない test well 用処理の Public LB は 11.740 で、ML route の既存 Public LB 基準だった exp026 12.102 を -0.362 更新した。一方、overall / PF route の exp027 public replay 基準 8.781 からは +2.959 悪化している。

結論として、ML route 基準は exp039 11.740 に更新する。全体 基準は exp027 8.781 のまま維持し、Ravaghi/PF 系は次に confidence / divergence feature や gate の材料として扱う。
