# タスクリスト

## TODO

- 初回runをdeterministic anchorとせず、独立rerunのposterior /
  prediction SHA一致後だけparity anchorを判断する。独立rerunは別承認が
  ある場合だけ行う。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 2026-07-29: `exp445_tvt_to_u_coordinate_parity_exact_hmm`として採番した。
- 2026-07-29: exp438のfixed absolute-U latticeと区別し、
  `U_t,j=P_j+Z_t`のrow-shifted coordinate relabelを固定した。
- 2026-07-29: candidate index transitionを親と同じ
  `r_current*delta_MD-delta_Z`に保つ代数contractを確定した。
- 2026-07-29: technical parityだけを判定し、CV / RMSE / LB / promotionを
  判定しない方針を確定した。
- 2026-07-29: `docs/06_reproducibility.md`を確認し、RNGなし、固定順、
  input / kernel / posterior / prediction SHAと独立rerun方針を記録した。
- 2026-07-29: 今回はsteering、backlog、実験scaffoldと記録文書だけを作り、
  実装・実行を行わない承認境界を記録した。
- 2026-07-29: 追加依頼「exp445を実装してください」により、
  compact self-contained train候補、inference禁止guard、専用testを実装した。
- 2026-07-29: parent TVTとcandidate row-shifted Uのemission、initial prior、
  physical position kernel、forward/backwardを別経路で組み立てた。
- 2026-07-29: exp209直接比較を追加し、position posteriorとlog-likelihoodの
  完全一致を確認した。coordinate expectationは双方を明示正規化し、
  exp209 raw matrix-product readoutもreport-onlyで保存する。
- 2026-07-29: synthetic variable-Z / constant-Z、physical edge、emission、
  prior、tiny brute-force、paired posterior/readout parity testを実装した。
- 2026-07-29: fixed32のcandidate 32 + paired parent 32 = 64 HMM well-runs、
  truth-free leakage ledger、coordinate / transition-emission / posterior /
  prediction SHA、deterministic gzip readback gateを実装した。
- 2026-07-29: 専用pytest 17件、exp438/exp445関連pytest 29件、
  Jupytext `--test`、py_compile、Ruff F821、strict experiment validationが
  PASSした。
- 2026-07-29: 親exp438 compact 2,780行 / 9 numbered sectionsに対し、
  exp445 compact trainは2,703行 / 9 numbered sectionsで、input、独立座標組立、
  exact HMM、brute-force、fixed32 freeze、gate、metricsをNotebook上に展開した。
- 2026-07-29: compact sourceの`__file__`と同一exp helper importは0件、
  正規`*_train.ipynb` / `*_inference.ipynb`は未変更。
- 2026-07-29: repository全体の`make test`は既存exp297 / 301 / 333 /
  336 / 349のconfig-contract不一致5件でcollection停止した。exp445専用17件と
  直接関連exp438込み29件は独立にPASSし、scope外の既存5実験は変更しなかった。
- 2026-07-30: 追加依頼「実行してください」により、正規Notebook採用、
  Kaggle package / push、fixed32 Stage 0の1回実行が承認された。
- 2026-07-30: 実行量をcandidate 32 + paired parent 32 = 64 HMM runs、
  reporting fold / LightGBM config / trained fold / booster / model / PF /
  Beam / GPUすべて0として再確認した。
- 2026-07-30: compact train / inference guardを正規Notebookへ採用し、
  metadata、source、loose / bootstrap config、fixed32 manifest SHAを監査した。
- 2026-07-30: Kaggle version 1はNumba初期化後の環境変数変更により
  HMM前にERROR。科学contractを変えず`set_num_threads(1)`だけへ修正した。
- 2026-07-30: Kaggle private CPU version 2（id_no `129095337`）を完了し、
  64 HMM well-runs、technical gate 16/16 PASS、
  `coordinate_parity_verified`を確認した。
- 2026-07-30: metricsとparity artifactを取得してraw / decompressed SHAを
  照合し、run flagを再ロックした。inference / submissionは未実施。
