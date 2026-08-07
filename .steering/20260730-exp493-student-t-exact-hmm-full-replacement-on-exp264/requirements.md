# 要件

## 依頼

exp374のStudent-t exact HMMを、exp264の元のGaussian `exact_hmm`と置き換える。
候補数を12本のまま維持してcorrected strict nested dual selectorを評価するための
バックログ、実験scaffold、steeringを作成し、実装前に設計を確定する。

## 制約

- Routeは`ensemble`。物理HMM候補とML selectorの両方が予測評価に本質的に寄与する。
- 親selectorは`exp264_exp263_candidate_confidence_dual_selector`、物理候補は
  `exp374_exp209_student_t_exact_hmm_emission`とする。
- 12候補のID、宣言順、primary 11本、fixed fallback 7本を変更しない。
- `exact_hmm` semantic slotだけをStudent-tへ置換し、依存する3 formula候補を再計算する。
- GaussianとStudent-tを共存させず、13本目として追加しない。
- exp264のcorrected Stage A 88列feature schemaを同じ名前・順序で再利用し、
  置換OOFを見て列をrefreezeしない。
- selectorは1 variant、2 objectives、outer 5 x inner 4、計40 CPU booster。
  保存済みexp264 controlは再学習しない。
- downstream TVT学習、current-test生成、inference、submissionはscope外。
- compact self-contained train候補の実装は2026-07-31のユーザー依頼で承認済み。
  canonical notebook採用、Kaggle package/push/runは別承認まで行わない。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- `candidate_contract.yaml`で12候補の順序、4 changed / 8 unchanged、formulaを一意に固定する。
- `feature_contract.yaml`で88列schema維持とdirect/transitive再計算範囲を固定する。
- exp374入力列allowlist、decompressed content SHA、global key join、truth-late順序を固定する。
- 技術・leakage・selector score・hard selectorのgateを実行前に固定する。
- 実行量を1 variant / 2 objectives / outer 5 / inner 4 / 40 CPU booster、
  control再学習0 / GPU 0 / downstream 0 / inference 0 / submission 0と明記する。
- compact self-contained train候補を別名で実装し、
  canonical train/inference notebookはmarkdown-only placeholderを維持する。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`へ
  compact候補実装済み・未実行の実験として記録する。

## 次の判断

canonical train notebookへの採用、Kaggle package、push、runは、
実装とは分けてユーザーの明示判断を受ける。
