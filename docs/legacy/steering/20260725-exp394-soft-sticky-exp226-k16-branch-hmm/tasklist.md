# exp394 タスクリスト

## TODO

- なし。

## ブロック中

- full 773-well OOFはfixed16 runtime gate FAILのため実行不可。

## 完了

- exp394 の目的、2 branch、GR 観測、K16 transition、soft-sticky transitionを固定した。
- 低ランク 3D 地層場を依存条件としないことを固定した。
- base switching length `1000 MD-ft`、docking `6.0 ft`、initial branch prior `0.5/0.5` を固定した。
- technical preflight は科学 score gate にしないことを固定した。
- full OOF の実行量、比較対象、promotion gate、禁止事項を固定した。
- backlog、steering、実験 scaffold、experiment summary を更新した。
- Jupytext percent形式の3,522行compact self-contained train sourceと別名Notebook候補を
  実装した。
- ユーザーの「実行してください」を、正規train採用、package/push、固定16-well
  technical preflightの承認として記録した。technical candidate 1 / 16 HMM well runs /
  LightGBM config・trained fold・booster・control rerun・GPU各0。full OOFは承認外。
- compact self-contained候補を正規train Notebookへ採用した。
- exp209全grid×41 rateを保持するE/H joint forward-backward、MD-aware hazard、
  E→H注入、H→E 6-ft docking、joint mean/std、branch posterior、expected switchを実装した。
- 固定16-well選択、runtime/RSS投影、prediction/branch/schedule SHA freeze、
  late truth/exp263/hidden-like join、promotion/recovery gateを実装した。
- dense全列挙parityを含む専用10 tests、Jupytext round-trip、py_compile、
  Ruff F821/E9をPASSした。
- canonical private CPU version 1（id_no `128536142`）でfixed16 preflightを完了した。
- 16/16 wells、finite/full-grid、identity/leakage、normalization、transition、
  RSSはPASSした。
- projected full runtime `112,736.889439 sec > 30,600 sec`だけがFAILし、
  `technical_blocker_not_scientific_negative_result`として閉じた。
- RMSE/CV/LB、full OOF、inference、submission、model/booster/control rerunは0。
