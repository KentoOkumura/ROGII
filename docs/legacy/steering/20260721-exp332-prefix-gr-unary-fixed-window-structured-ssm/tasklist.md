# タスクリスト

## 未着手（別承認が必要）

- なし。Stage 0 runtime gate FAILにより後続は閉鎖。

## 進行中

- なし。

## ブロック中

- Stage A/B/C、推論、提出はStage 0 runtime gate FAILにより閉鎖。

## 完了

- [x] exp332をfixed-window structured training案として別採番した。
- [x] 256 rows、3 windows/well/epoch、teacher boundary、soft structured objectiveを固定した。
- [x] full-well evaluationとexp295同等promotion gateを固定した。
- [x] reproducibility/SHA/runtime gateを固定した。
- [x] design-only experiment scaffoldを作成した。
- [x] exp331がStage A科学gate FAILでbranch close済みであることを確認した。
- [x] ユーザー依頼「exp332を実装してください」によりimplementation-only scopeの承認を得た。
- [x] 256-row/3-window deterministic manifestとteacher-boundary contractを実装した。
- [x] compact self-contained Jupytext train候補とfail-closed inference候補を実装した。
- [x] window selection、truth非参照、boundary非入力、4-sweep gradient、full-well evaluationの専用testを作成した。
- [x] ユーザー依頼「実行してください」により固定16-window T4 Stage 0だけの実行承認を得た。
- [x] Kaggle T4 version 1の固定16-window Stage 0を完走した。
- [x] memory `1.203263 GB`はPASS、保守的runtime `13.151137 h`はFAILと判定した。
- [x] report/selection/boundary/measurement/log SHAを記録し、事前契約どおりbranchを閉じた。
