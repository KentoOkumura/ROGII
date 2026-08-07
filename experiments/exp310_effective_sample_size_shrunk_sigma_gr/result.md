# exp310_effective_sample_size_shrunk_sigma_gr 結果

## 状態

設計確定済みだが、exp307 promotion gate FAILにより未実装・未実行のまま閉鎖。

## 固定した検証契約

- 変更対象: finite-MAD `σ_GR`の有効標本数依存shrinkageだけ。
- trigger FAIL: HMMを実行せずclose。
- trigger PASS: 1 variant、最大773 HMM well-runs。
- promotion: exp307比`>=0.03 ft`改善、4/5 folds、long-tail/hidden-like/by-well/fixed-blend guard全PASS。

## 結果

CV、LB、生成物、SHAはまだ存在しない。設計確定は実行承認を意味しない。

## 次

exp307 PASSの必須条件が成立しないためtriggerを評価せず、HMM、inference、submissionを行わない。
