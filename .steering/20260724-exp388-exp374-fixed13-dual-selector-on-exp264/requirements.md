# 要件

## 依頼

exp374 の固定 `df=4` Student-t absolute-TVT exact-HMM train-side予測を、
corrected exp264 candidate-long dual selector の fixed12 bank に13本目として追加し、
候補単独の平均改善を selector がwell/row contextから安全に利用できるか検証する。

## 制約

- Route: `ensemble`
- 親 selector は `exp264_exp263_candidate_confidence_dual_selector` とし、
  fixed12の候補、目的関数、exp263 selector fold、sample cap、LightGBM設定、
  fixed fallback 7候補を変更しない。
- 変更変数は primary hard-select domain へ
  `student_t_exact_hmm` を1候補追加することだけとする。
- exp374のtail gate失敗をPASSへ再分類しない。候補単独採用、blend、fallback、
  threshold tuning、同一OOF上の救済を行わない。
- exp374生成物はtarget-free列
  `well_id,row_idx,student_t_*_hmm_tvt,student_t_*_hmm_std,
  student_t_*_hmm_loglik`だけを読み、`id`はkey整合監査専用とする。
- exp374はwell間学習を持たない決定的HMM候補なのでsource foldは特徴に作らず、
  global `(well_id,row_idx)` join後にexp263 selector foldへpartitionする。
- selector fitはCPUのみ。active variant 1、objective 2、outer fold 5、
  inner fold 4、合計40 booster。親controlの再学習、GPU学習、downstream TVT、
  inference、submissionは行わない。
- 再現性は `docs/06_reproducibility.md` に従い、exp374 gzipのraw SHAと
  decompressed content SHA、selector feature/model/prediction SHA、
  Kaggle kernel versionを記録する。

## 受け入れ基準

- exp374入力が3,783,989行、773 wells、key重複0、有限率1.0で、
  expected decompressed SHAと一致する。
- fixed12 + Student-t候補のfixed13 contract、primary 12候補、
  fixed fallback 7候補、compact 77特徴が固定される。
- exp374のtarget/error列読込0、source fold特徴0、global key join欠損0を
  manifestとテストで確認する。
- 40/40 selector modelが生成され、parent fixed12 scoreとのpaired readout、
  candidate usage、fold/scope/by-well tail、post-freeze noveltyを保存する。
- selector科学gateはpooled非悪化、4/5 folds改善、利用率、scope/tailを
  事前固定し、失敗時は追加調整せずfail-closeする。
- gzip生成物はraw `.csv.gz` SHAではなくdecompressed content SHAを
  主証拠として記録する。
