# exp381_formation_contact_order_semimarkov_hmm

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU Stage 0 FAIL・branch closed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: `exp209_emission_dynamics_direct_hmm`

## 仮説

train-only地層列と正解TVTの関係は、局所rateとして移送するより、
坑井が6地層面を横切る接触位置・contact-TVT・順序として表現した方が、
exp209 exact HMMへ有効な物理制約を与える。

## 変更点

- outer-trainだけの`FormationPlaneKNN(k=10)`で6地層面を対象坑井へ外挿する。
- `Z - F_hat = 0`のMD昇順first crossingをformationごとに1個だけ取る。
- outer-train真接触のcontact-TVT中央値を、target known prefixの単一offsetで校正する。
- outer-train formation中央値のconstant surfaceをpaired controlにする。
- surface / crossing / contact予測をSHA freezeしてからvalidation truthをlate joinする。
- Stage 0が不合格なら7-state semi-Markov HMMを実装しない。

## 検証方針

- Fold: exp209系が比較に使う保存済みexp226 outer 5-fold identity
- Group: well
- Stage 0: eligible率、contact event数、crossing MD MAE/p90、
  contact-TVT RMSE、fixed formation order率、constant比gain、fold安定性
- Leakage Check: outer-validの生Formation / TVTはtarget-free freeze前にread 0
- Resource: SHA256順の固定16坑井、report-only
- Stage 1: Stage 0 FAILのため未実装・0 runで閉鎖

## 実装

- `*_compact_selfcontained_train.py` / `.ipynb`に0-HMM Stage 0を実装済み。
- trainはfold identity、read guard、surface、crossing、prefix offset、SHA freeze、
  truth late join、固定AND gate、生成物保存を10章で展開する。
- `*_compact_selfcontained_inference.py` / `.ipynb`はStage 1未実装を確認してfail closedする。
- compact trainを正規train Notebookへ採用した。正規inference scaffoldは未採用。
- 専用test 10件、Ruff、py_compile、Jupytext round-trip、strict実験validationをPASSした。

## 実行入口

- 正の実装はcompact self-contained sourceと、それから生成した正規train Notebook。
- Kaggle private CPU version 1は科学処理前ERROR、formation別finite donor修正後の
  version 2を完了した。
- Stage 0実行量はdiagnostic 1 / surfaces 6 / folds 5 /
  model・HMM・PF・Beam・booster各0 / parent control再実行0 / GPU 0。
- inference / submissionは利用不可。

## 結果

| メトリック | 値 |
| --- | --- |
| 実装検証 | PASS |
| 専用test | 10 passed |
| Stage 0 | FAIL（contact-TVT RMSEのみ不合格） |
| eligible / events | `349/773` / `1,291` |
| crossing MD MAE / p90 | `35.994405 / 61.799226 ft` |
| contact-TVT RMSE | `44.770101 ft` |
| order / positive folds | `0.997135` / `5/5` |
| Stage 1 / CV | 未実装 |
| Public LB | - |

## 所見

### 良かった点

- formation-relative K16とは独立の接触仮説を、0-HMMで先に棄却できる形にした。
- target生Formation、suffix truth、oracle crossing、post-hoc order修復をコードで禁止した。

### 悪かった点

- contact-TVT RMSEは`44.770101 ft`で上限15 ftを大きく超えた。

### リスク / 注意

- multiple crossing、地層面の局所逆転、接触不足、surface外挿誤差が主要リスク。
- surface k、crossing選択、formation除外、offset、gateの実行後救済は禁止。

## 次

固定AND gateに従いbranchを閉じる。surface / offset / gate救済、7 ordered interval
semi-Markov HMM、inference、submissionは実行しない。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、
実験名や設定名を除いて日本語優先で記録する。
