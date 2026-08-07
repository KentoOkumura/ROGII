# exp381_formation_contact_order_semimarkov_hmm 結果

## 仮説

地層接触順序と区間durationはexp209 HMMへ有効な物理制約を与える。

## 設定

- 親: exp209
- 検証: 保存済みexp226 outer 5-fold identityを使う0-HMM contact readout
- 主surface: outer-train `FormationPlaneKNN(k=10)`
- 対照: outer-train formation中央値constant surface
- crossing: full-well、formation別first crossing、線形補間
- target校正: known prefixから坑井単一additive offset
- メトリック: crossing MD MAE/p90、contact-TVT RMSE、order率、constant比gain
- 実行量: model / HMM / PF / Beam / booster各0

## 実装結果

| 項目 | 結果 |
| --- | --- |
| compact self-contained train | 実装済み |
| fail-closed inference | 実装済み |
| 専用test | 10 passed |
| Ruff / py_compile | PASS |
| Jupytext round-trip | PASS |
| strict experiment validation | PASS |
| 正規train Notebook採用 | 実施 |
| Kaggle Stage 0 | private CPU version 2完了 |

## 科学結果

| メトリック | 値 |
| --- | --- |
| eligible well率 | `349/773 = 0.451488`（PASS） |
| contact event数 | `1,291`（PASS） |
| crossing MD MAE / p90 | `35.994405 / 61.799226 ft`（PASS / PASS） |
| contact-TVT RMSE | `44.770101 ft > 15 ft`（FAIL） |
| 正しい接触順率 | `0.997135`（PASS） |
| constant surface比gain | `687.676085 ft`（PASS） |
| 改善fold | `5/5`（PASS） |
| CV / Public LB | - / - |

## 再現性

- seed policy: no RNG、stable fold/well/formation/MD order
- validation truth / formation pre-freeze read: `0 / 0`をmanifestで確認
- surface / crossing / contact / resource SHA: 保存・検証済み
- gzip: decompressed content SHAを主証拠にする
- model / prediction / submission SHA: fitted modelなし、crossing readoutは実行時、
  submissionは対象外
- target-free bundle logical SHA: `69787c31...4b6955`
- truth manifest logical SHA: `b27a1c42...0cf2c`
- contact event logical SHA: `c967ee57...12c11`
- 期待15 artifactが存在し、SHA manifest 14行のraw / decompressed SHAは全一致
- deterministic anchor: 初回runのみのためfalse

## 解釈

地層面の幾何学的接触位置はMAE `35.99 ft`、順序率`99.71%`、constant surface比
`687.68 ft`改善と強く、formation surfaceの位置・順序移送自体は成立した。一方、
contact-TVTは全foldで上限15 ftを超え、pooled `44.77 ft`だった。したがって、
known-prefix単一offsetで校正したcontact centerをsemi-Markov duration priorへ
昇格する事前条件は満たさない。固定AND gateに従いStage 1を実装せずbranchを閉じる。

## 次

surface k、formation除外、offset、gateのpost-hoc救済は行わない。Stage 1、
inference、submissionも実行しない。位置・順序の知見は別仮説で再利用する場合も
新しい事前設計と独立した根拠を必要とする。
