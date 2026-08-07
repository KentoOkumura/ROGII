# exp081_pilkwang_branch_decomposition

## 状態

- ルート: pf_beam
- 状態: decomposition_completed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-19
- 親実験: exp079_public_artifact_replay_integrity_audit

## 仮説

Pilkwang final は projected ridge/PF、pretrained LGBM、model-package tiny gate の合成だが、final 全体を直接 submit する前に branch 単位の距離と risk を分解すれば、提出回数を使う候補を 1-2 個に絞れる。

## 変更点

- `pilkwang_branch_decomposition.py` を追加し、exp079 v4 の summary / submission summary / pairwise JSONL を二次解析する。
- branch role、final 差分、ridge-sp 差分、candidate decision、submit candidate rank を artifacts に保存する。
- train / inference notebook を同じ audit entrypoint に更新した。

## 検証方針

- Fold: なし。target-free pairwise audit。
- Group: なし。
- Stratification: なし。
- Leakage Check: 新規学習・提出なし。exp079 の risk hits を引き継ぎ、row-level diff が未保存であることを明示する。

## 実行入口

- 学習 notebook: `exp081_pilkwang_branch_decomposition_train.ipynb`
- 推論 notebook: `exp081_pilkwang_branch_decomposition_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp081_pilkwang_branch_decomposition`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Candidate count | 16 |
| Submit candidates | 2 |

## 所見

### 良かった点

- `submission_projected_ridge_pf_projection_d4_b075_raw.csv` が ridge-sp に最も近い Pilkwang branch だった。final との差は RMSE 1.442298、ridge-sp との差は RMSE 1.130190。
- `submission_projected_ridge_pf_pretrained_lgbm_w0.60.csv` は final から RMSE 0.144364 と近く、ridge-sp 方向へ少し動くため、2 番手の submit 検討候補にした。

### 悪かった点

- pretrained LGBM 単独は ridge-sp との差が RMSE 3.205317 で、独立 submit 候補としては弱い。
- model-package-only は final との差が RMSE 17.318521 と大きく、diagnostic only。

### リスク / 注意

- exp079 v4 のローカル output は候補 CSV 本体を保存していないため、row-level segment guard は未実施。
- Pilkwang notebook は `exact_match_or_override=38`、`writes_submission_csv=3` の risk hits がある。現 profile では exact / override 無効前提だが、改善根拠にはしない。
- exp027 / exp073 / exp063 anchor との pairwise は exp079 に保存されておらず、今回の anchor comparison では `missing_pairwise` として扱う。

## 次

- submit するなら上位 2 件だけに絞る前提で、候補 CSV 本体を再取得して submit-check / row-level guard を行う。
- SP45 / fle3n / Koolbox 系の exact source slug を固定して追加監査へ進む。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
