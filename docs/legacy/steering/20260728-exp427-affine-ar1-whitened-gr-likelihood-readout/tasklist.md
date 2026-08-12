# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- technical / scientific gate FAILのため、rerun、HMM / PF decoder、inference、
  submissionはterminal close。

## 次のアクション

- exp427を再開しない。独立した必要性とユーザー承認がある場合だけ、保存生成物を
  入力とする低優先度P4 `affine_ar1_rank_failure_attribution_readout`を別steeringへ
  切り出す。目的は失敗原因の説明に限定し、parameter探索や昇格に使わない。

## 完了

- exp209 / 280 / 343 / 345 / 359 / 360 / 374 / 389の関連結果を確認した。
- `docs/06_reproducibility.md`を読み、再現性契約へ反映した。
- exp425の既存steering予約とexp426の既存backlog / steering予約を避け、
  exp427を採番した。
- prefix affine posteriorとfold-safe AR(1) predictive likelihoodの数式を固定した。
- `identity/affine × iid/AR1`の2×2要因分解を固定した。
- block、shift、support、gate、negative control、禁止事項を固定した。
- Stage 0が0-HMM / 0-PF / 0-modelであることを固定した。
- steeringとdesign-only実験scaffoldを作成した。
- `KAGGLE_DIRECTION.md`へ低-中P3として追加した。
- `make validate-exp EXP=exp427_affine_ar1_whitened_gr_likelihood_readout`を
  strict PASSした。
- `make validate-template`をPASSした。
- `make update-summary`で`experiment_summary.md`を423実験へ更新した。
- `review_exp_docs.py exp427 --root .`でcore evidence categoriesが揃うことを確認した。
- ユーザーの追加依頼をimplementation承認として記録した。
- compact self-contained Jupytext train候補を実装した。
- affine posterior、fold-safe AR1、4 score、saved exp280 alignment、
  truth-late readout、全AND gateのcontract testを作成した。
- prediction / submissionを拒否するfail-closed inference候補を実装した。
- Jupytext train / inference変換とround-trip testをPASSした。
- py_compile、Ruff、専用pytest 14件をPASSした。
- 親exp280の9章・1,165行に対し、exp427 train候補は12章・約2,200行で、
  同一実験helper importのないself-contained構成であることを確認した。
- `make validate-exp EXP=exp427_affine_ar1_whitened_gr_likelihood_readout`と
  `make validate-template`を再PASSした。
- ユーザーの実行依頼を、正規train Notebook採用、Kaggle package / push、
  固定Stage 0 CPU runの承認として記録した。
- 54文字の初回slugはKaggle `SaveKernel 400`、直後pull 403で未作成と確認した。
  科学契約を変えず、43文字の
  `exp427-affine-ar1-whitened-gr-readout-train`へid/titleを同時短縮した。
- version 1はprefix posterior 773/773 wells後、最初のscore対象外wellの
  schemaなし空DataFrame sortでERROR。科学条件を変えない実装修正へ限定した。
- score / negative controlへ型付き空schemaを追加し、専用回帰testを含む
  15 tests、Jupytext、py_compile、Ruff、strict validationをPASSした。
- Kaggle private CPU version 2（id_no `128931242`）で773 wells、7,787 blocksを
  完走した。runtime `4,358.768411秒`、peak RSS `1.264053 GB`。
- eligible block率`0.721074 < 0.75`でtechnical FAIL。primary
  `affine_ar1`はMRR / top3 `0.386090 / 0.439181`で、matched
  `0.388003 / 0.450401`、saved exp280 `0.388146 / 0.449866`の双方を下回った。
- `stage_0_failed_close_without_rescue`を記録し、exp427と条件付きexp431を
  backlogから削除した。新規P4は保存生成物だけの失敗原因分解に限定した。
