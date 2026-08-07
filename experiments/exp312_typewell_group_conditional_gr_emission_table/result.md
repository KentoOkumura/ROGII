# exp312 結果

## 状態

Kaggle private CPU version 1を完了し、固定promotion gateはFAILした。branchは救済なしで閉じる。

## 仮説と変更点

exp311のgroup residual平均gainを、固定deployable12候補に対する条件付きGR emissionの順位改善へ変換できるかを検証する。親から直接affine補正を持ち込まず、global-unconditional Student-tをbaselineに、group/GR decile/gradient/missingness tableだけを追加する。

## 実装内容

- exp293 deployable12の12候補をexp263 manifestから固定順・固定formulaで再構成する。
- exp311のfoldと`native_overlap_1` group membershipをSHA固定入力として使う。
- outer-trainだけでType Well GR decile × |gradient| tertile × horizontal欠損flagのStudent-t tableを作る。
- baseline、real、group shuffle、well内candidate-TVT shiftの4順位を凍結してからouter-valid TVTを開く。
- candidate生成、HMM/PF/Beam decode、ML、inference、submissionは行わない。

exp311の全gate条件はユーザー判断で上書きしたが、fit-RMSE R²とworst-well FAILは入力契約とsummaryへ残す。

## 静的検証

Jupytext train/inference、構文、ruff、専用9 tests、親を含む16 tests、strict experiment/template validationはすべてPASSした。compact trainを正規notebookへ採用後にも同じ検証を再実行してPASSした。

## Kaggle実行

- Kernel: `kentookumura/exp312-typewell-gr-emission-rank-audit-train` version 1、id_no `128090149`。
- private CPU、internet off、runtime `326.622947 sec`。
- scientific 1 + controls 2、5 folds、model/booster/decoder `0/0/0`。
- deployable12は3,783,989 rows / 773 wells、候補生成run 0、formula parity PASS。
- 全foldで順位freeze前のouter-valid truth accessは0。

## 結果

| 指標 | baseline | conditional real | 差 |
|---|---:|---:|---:|
| MRR | 0.336112 | 0.334519 | -0.001592 |
| top3 rate | 0.358090 | 0.355646 | -0.002444 |

- 改善fold: `0/5`。
- real minus group-shuffle MRR: `+0.001611`。
- real minus candidate-TVT shift MRR: `+0.063809`。
- hidden-like nonregression: 2面ともFAIL。
- fallback率: `1.823%`でPASS。
- 生成物: 予定10/10件、raw SHA整合9/9件。

TVT-row alignmentを壊すshift controlには差がある一方、group-label shuffleとの差はほぼなく、global-unconditional Student-tより平均順位も悪化した。したがって、候補TVT上のGR evidence自体は存在しても、今回のType Well群×decile×gradient×missing条件づけには追加価値がない。

失敗原因は、exp311で見えた平均的な群noise共通性がcandidate間の微細な順位識別には弱く、条件セル分割後も群ラベル固有の情報がほぼ残らなかったことだと考える。group-shuffle差が`+0.001611`に留まることがこの解釈と整合する。

## 固定した判定

MRR、top3、real-vs-shuffled、4/5 folds、hidden-likeの5項目をFAILし、fallbackとlate-truthだけPASSした。事前契約どおりbin/df/shrinkageを救済しない。

## 次

exp312 branchを閉じ、exp313〜320は停止を維持する。このFAILを根拠に同じ条件軸の救済実験は追加しない。回避策として、群priorに依存せずexp304で直接支持されたwithin-well SWT signalを使う既存exp305を次候補とし、その後は既存順序どおり独立なexp321 Z-only構造監査へ戻る。
