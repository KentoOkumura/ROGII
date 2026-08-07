# exp335_signed_residual_meta_on_exp264

## 状態

- Route: `ml_model`
- Status: Stage D固定tail guard FAILを保持、CPU inference / submit-check完了、Public LB `7.517`
- Downstream OOF RMSE: `8.146107755881022`
- Saved exp264: `8.460811237612477`（`0.314703 ft`改善）
- Scientific-support / promotion: FAIL / FAIL
- Public / Private LB: `7.517` / 未公開
- Inference: Kaggle CPU version 3 COMPLETE / user-submitted ref `54928806` COMPLETE
- 親: `exp264_exp263_candidate_confidence_dual_selector`

## 仮説

exp264の既存selector compact 74列を維持し、12候補のstrict-nested signed residual方向情報23列をadd-onlyすると、後段TVT LightGBMを改善できるか検証した。

## 変更点

単一変更は`true_tvt - candidate_tvt`を予測する23列の追加だけである。clean273、saved74、fold、seed、候補順、downstream 3 configを固定し、saved controlを再学習していない。

## 検証方針

exp264と同じ5 outer foldsのOOFでpooled/fold、距離帯3面、hidden-like 2面、by-well tailを比較した。scientific-supportは平均改善だけでなくby-well p95とworstをAND条件とし、promotionはclean273比worstと`+1/+3/+5 ft`悪化well数の非増加も要求した。

## 実行量

- Stage S: outer 5 × inner 4、20 CPU boosters、PASS
- Stage D: 1 variant × 3 configs × 5 folds、15 GPU boosters、完了
- 特徴: clean273 + saved74 + signed23 = 370列
- saved control再学習: 0
- Kernel: `kentookumura/exp335-signed-residual-meta-on-exp264-tvt-train` version 2
- Runtime: Kaggle T4 / internet off、約5時間33分37秒

## 結果

pooled RMSEは`0.314703 ft`改善し、4/5 foldsと全5 scopeでsaved exp264を非悪化だった。しかしby-well delta p95は`+1.728657 ft`、worst-well deltaは`+10.238752 ft`で固定上限を超えた。clean273比のworst deltaと`+1/+3/+5 ft`悪化well数もすべて増えたため、promotionしない。

signed 23列は非ゼロgainを持ち、単一特徴への極端な集中もなかった。方向signal自体は有効だが、well-tailへ安全に一般化しないという結論である。

## 所見

平均RMSEの改善は4 foldsと全scopeに広がっており偶然の単一fold依存ではない。ただし悪化側の裾が大きく、事前固定した採用条件では安全でない。後付けthresholdで同じOOFを救済せず、negative promotion resultとして保持する。

## 生成物

- Stage D small artifacts: `kaggle/output/stage_d_v2/artifacts/`
- Kernel log: `kaggle/output/stage_d_v2/exp335-signed-residual-meta-on-exp264-tvt-train.log`
- Inference small artifacts: `kaggle/output/inference_v3/artifacts/`
- Inference log: `kaggle/output/inference_v3/exp335-signed-residual-meta-on-exp264-inference.log`
- 詳細: `result.md`、`metrics.json`、`SESSION_NOTES.md`

## 終了判断

事前登録どおりgateを緩和せず、同一実験でsigned target/objective/grid/threshold/特徴を救済しない。2026-07-23のユーザーoverrideにより保存済みmodelのCPU inferenceとsubmission file生成だけを例外実施し、version 3で14,151行の出力とsubmit-check PASSを得た。ユーザーによるcode submissionはPublic LB `7.517`を記録した。

## 次のアクション

非promote判断を維持し、Public LB `7.517`をsubmitted reference anchorとして扱う。
