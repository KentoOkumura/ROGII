# タスクリスト

## 目的

exp209の観測モデルを固定し、well別`sig_r,w`だけを変更するexp338を、実装・実行承認と後続分岐条件を混同せず進める。

## 進行中

- なし。exp338はversion 3のpromotion gate FAILによりterminal close。

## ブロック中

- inference/submissionはpromotion gate FAILにより実施しない。
- 新exp323相当と新exp324--327相当は先行PASS条件不成立のため作成しない。

## 完了

- 2026-07-22: `exp338_exp209_well_adaptive_transition_noise`として採番し、steeringと実験scaffoldを作成した。
- 2026-07-22: 科学的親をexp209へ固定し、旧exp309はtransition-scale式だけの参照元に限定した。
- 2026-07-22: well別`sig_r,w`式、fallback、shrink、clip、実行量、promotion gate、truth-freeze、SHA方針を固定した。
- 2026-07-22: exp338 PASS後の新exp323相当、新exp323 PASS後の新exp324--327相当という採番・承認依存を固定した。
- 2026-07-22: 旧exp323--328を再開・reparentしないことを固定した。
- 2026-07-22: 旧exp328仮説の独立再検証入口を`exp345_exp209_time_varying_gr_affine_calibration_hmm`へ固定し、exp338 chainと相互非依存にした。
- 2026-07-22: 初回design-only段階では実装、Notebook編集、Kaggle package/push/run、inference、submissionを行わなかった。
- 2026-07-22: 別実装承認により、compact self-contained train候補へknown-prefix transition auditとexp209-compatible exact-HMM candidateを実装した。
- 2026-07-22: saved exp209 HMM/LikPF、fold、hidden-likeのSHA/header preflight、prediction/audit freeze後のlate truth join、全promotion gateを実装した。
- 2026-07-22: fail-closed inference候補と専用contract testを追加し、exp209 synthetic observation/state parityを確認した。
- 2026-07-22: Jupytext変換/test、py_compile、ruff F821、専用pytestを完了し、実行承認後にcompact self-contained trainを正規Notebookへ採用した。
- 2026-07-22: `make validate-exp` strictと`make validate-template`をPASSした。全体pytestは605 passed / 3 skippedで、既存exp296の状態と旧test期待値の不一致2件だけがFAILした。
- 2026-07-22: version 1は親Kaggle raw metricsとローカル追記後metricsのschema差でHMM前ERROR。raw metrics SHAとnested parityを検証する契約へ修正した。
- 2026-07-23: version 2は773/773 HMM完走後、exp115の正式な`purged_train_excluded`をlate role契約が拒否してERROR。artifact SHA、列別role許容値、件数を固定した。
- 2026-07-23: 実exp115 artifactを使う回帰testを追加し、専用test 12件、Jupytext、py_compile、ruff、strict validationをPASSした。
- 2026-07-23: 1 variant / 773 HMM / model・booster・PF・Beam・control再実行0を再確認し、同じcanonical kernelへversion 3をpushした。push直後はRUNNING。
- 2026-07-23: version 3は773/773 HMM、3,783,989 rowsを11,376.512秒で完了した。
- 2026-07-23: direct `14.062348` vs parent `11.938287`、0/5 folds、fixed LikPF blend `11.184022` vs `10.269693`で科学gateをFAILした。
- 2026-07-23: 全773 wellsが`sig_r=0.004`へhigh clipされ、clip fraction `1.0`でtechnical gateもFAILした。
- 2026-07-23: decision `adaptive_sig_r_failed_close_without_rescue`を記録し、inference、submission、後続chainなしでterminal closeした。

## 次のアクション

なし。transition-noise適応を将来独立に再訪する場合だけ、HMM前のtarget-free proxy identifiability preflightを別途設計する。
