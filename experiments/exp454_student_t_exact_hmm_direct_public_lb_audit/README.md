# exp454_student_t_exact_hmm_direct_public_lb_audit

## 状態

- ルート: `pf_beam`
- 状態: 設計確定、実装未着手
- train-side OOF: `11.720478702`
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-30
- 親実験: `exp374_exp209_student_t_exact_hmm_emission`

## 仮説

固定`df=4.0` Student-t emissionはGaussian exact HMMよりoverall、4/5 folds、
全固定scopeを改善した。大きなwell-tail FAILがPublic LBでどの程度現れるかを確認する。

## 変更点

- 新しいHMMは作らない。
- exp374のStudent-t posterior meanを単体の`tvt`として出力する。
- exp209 HMMのgrid、rate、transition、prior、sigma、GR補完を変更しない。
- Gaussian/Huber control、blend、selector、gate、postprocessを実行しない。

## 検証方針

- Fold: train-side evidenceは既存5-foldを読み取り専用で参照
- Group: `well_id`
- Public LB: exp434 `exact_hmm` direct LBと比較
- Leakage Check: suffix TVT/error/fold/hidden-like roleをdecode入力にしない
- 再現性: no RNG、固定well/row/grid/rate順、source/config/prediction/submission SHA

## 実行入口

- 将来の評価Notebook:
  `exp454_student_t_exact_hmm_direct_public_lb_audit_inference.ipynb`の1本
- train Notebookはtemplate scaffoldのみで、実行・実装対象ではない
- Kaggle準備・実行・提出はいずれも別承認まで禁止

## 結果

| メトリック | 値 |
| --- | ---: |
| Gaussian exact HMM OOF | 11.938287235 |
| Student-t exact HMM OOF | 11.720478702 |
| OOF改善 | 0.217808533 |
| 改善fold | 4/5 |
| by-well delta p95 | +0.982661344 |
| worst-well delta | +35.015963236 |
| Public LB | - |

## 所見

### 良かった点

- overall、raw observed/missing、high-missing、long-tail、hidden-likeを改善した。
- OOF改善はHuber候補より大きい。

### 悪かった点

- 1 foldが悪化し、p95とworst wellのtail guardを大幅に超えた。

### リスク / 注意

- 平均LBが改善してもtrain-side tail FAILを取り消さず、自動採用しない。
- exp434 exact HMM LBを見てdfやscaleを変更しない。

## 次

- ユーザーが別途承認した場合だけ、設計どおり1本のinference Notebookを実装する。
