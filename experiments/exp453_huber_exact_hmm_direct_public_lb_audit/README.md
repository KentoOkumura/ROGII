# exp453_huber_exact_hmm_direct_public_lb_audit

## 状態

- ルート: `pf_beam`
- 状態: 設計確定、実装未着手
- train-side OOF: `11.852741130`
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-30
- 親実験: `exp389_exp209_huber_exact_hmm_emission`

## 仮説

固定`delta=1.345` Huber emissionはGaussian exact HMMより5/5 foldsと
全固定scopeを改善した。小さいwell-tail FAILがPublic LBでは顕在化しない可能性がある。

## 変更点

- 新しいHMMは作らない。
- exp389のHuber posterior meanを単体の`tvt`として出力する。
- exp209 HMMのgrid、rate、transition、prior、sigma、GR補完を変更しない。
- Gaussian/Student-t control、blend、selector、gate、postprocessを実行しない。

## 検証方針

- Fold: train-side evidenceは既存5-foldを読み取り専用で参照
- Group: `well_id`
- Public LB: exp434 `exact_hmm` direct LBと比較
- Leakage Check: suffix TVT/error/fold/hidden-like roleをdecode入力にしない
- 再現性: no RNG、固定well/row/grid/rate順、source/config/prediction/submission SHA

## 実行入口

- 将来の評価Notebook:
  `exp453_huber_exact_hmm_direct_public_lb_audit_inference.ipynb`の1本
- train Notebookはtemplate scaffoldのみで、実行・実装対象ではない
- Kaggle準備・実行・提出はいずれも別承認まで禁止

## 結果

| メトリック | 値 |
| --- | ---: |
| Gaussian exact HMM OOF | 11.938287235 |
| Huber exact HMM OOF | 11.852741130 |
| OOF改善 | 0.085546105 |
| 改善fold | 5/5 |
| by-well delta p95 | +0.002234351 |
| worst-well delta | +1.750248202 |
| Public LB | - |

## 所見

### 良かった点

- overall、5/5 folds、missing、long-tail、hidden-likeを一貫して改善した。
- p95悪化は`0.002234 ft`と非常に小さく、tail guardの保守性を検証しやすい。

### 悪かった点

- worst wellは`+1.750248 ft`悪化し、固定上限`+0.25 ft`を超えた。

### リスク / 注意

- OOF gainが小さいためPublic LB表示3桁ではtieになる可能性がある。
- exp434 exact HMM LBを見てdeltaやscaleを変更しない。

## 次

- ユーザーが別途承認した場合だけ、設計どおり1本のinference Notebookを実装する。
