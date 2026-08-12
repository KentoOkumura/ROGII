# 要件

## 依頼

exp389のHuber exact HMMを、exp264の元のGaussian `exact_hmm`と置き換える。
候補数を12本のまま維持してcorrected strict nested dual selectorを評価するための
バックログ、実験scaffold、steeringを作成して設計を確定し、2026-07-31の
ユーザー依頼に基づきcompact self-contained train候補と専用testを実装し、
canonical採用後にKaggle private CPUで固定Stage A/Cを実行する。

## 制約

- Routeは`ensemble`。物理HMM候補とML selectorの両方が予測評価に本質的に寄与する。
- 親selectorは`exp264_exp263_candidate_confidence_dual_selector`、物理候補は
  `exp389_exp209_huber_exact_hmm_emission`とする。
- 12候補のID、宣言順、primary 11本、fixed fallback 7本を変更しない。
- `exact_hmm` semantic slotだけをHuberへ置換し、依存する3 formula候補を再計算する。
- GaussianとHuberを共存させず、13本目として追加しない。
- exp264のcorrected Stage A 88列feature schemaを同じ名前・順序で再利用し、
  置換OOFを見て列をrefreezeしない。
- selectorは1 variant、2 objectives、outer 5 x inner 4、計40 CPU booster。
  保存済みexp264 controlは再学習しない。
- downstream TVT学習、current-test生成、inference、submissionはscope外。
- compact self-contained実装、canonical notebook採用、Kaggle package/push/runは
  2026-07-31のユーザー依頼で承認済み。追加rerunは別承認まで行わない。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- `candidate_contract.yaml`で12候補の順序、4 changed / 8 unchanged、formulaを一意に固定する。
- `feature_contract.yaml`で88列schema維持とdirect/transitive再計算範囲を固定する。
- exp389入力列allowlist、decompressed content SHA、global key join、truth-late順序を固定する。
- 技術・leakage・selector score・hard selectorのgateを実行前に固定する。
- 実行量を1 variant / 2 objectives / outer 5 / inner 4 / 40 CPU booster、
  control再学習0 / GPU 0 / downstream 0 / inference 0 / submission 0と明記する。
- 別名Jupytext source / ipynb候補、共有replacement helper、専用testが存在し、
  採用済みcanonical train notebookが同じStage A/Cを実行する。
- compact候補は候補順、4 changed / 8 unchanged、formula、global key join、
  truth-late、固定88/74 schema、科学gate、SHAをfail-closedで検証する。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`へKaggle結果とclose判断を記録する。

## 次のアクション

scientific gate FAILに従いbranchを閉じる。post-readout importanceバグは
canonicalで修正するが、追加40 boosterのrerunは別承認なしに行わない。
