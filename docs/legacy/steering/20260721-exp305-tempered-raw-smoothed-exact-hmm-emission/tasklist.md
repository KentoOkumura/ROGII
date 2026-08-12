# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp304 reserved follow-up案2を`exp305_tempered_raw_smoothed_exact_hmm_emission`として採番した。
- requirements/designで入力SHA、固定beta、HMM contract、保存済みcontrol、評価順序、成功条件、禁止事項を固定した。
- 実行量を1 scientific variant、773 HMM well-runs、model/LightGBM/PF/Beam/booster 0、control再実行0として固定した。
- `docs/06_reproducibility.md`に沿って大容量gzip、prediction content SHA、Kaggle bootstrap、thread/runtime記録を設計した。
- 設計段階では実装と実行を停止し、追加承認後に実装だけを開始した。Kaggle package/push/run、inference、submissionは行っていない。
- compact self-contained Jupytext trainを実装し、入力preflight、exact-HMM、prediction SHA freeze、late truth/control join、metrics、生成物保存をNotebookセル上へ展開した。
- fail-closed inference source/Notebookを実装し、raw-test predictionとsubmission生成を禁止した。
- exp304 series/manifest/summary/contract/input manifest、exp209 HMM/exp072 cache、exp226 fold、exp115 hidden-like assignmentのresolverとSHA guardを実装した。
- 固定`ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`、共有raw sigma、exp209 HMM contract、posterior normalization、truth-freeze、inference guardのsynthetic/unit testを追加した。
- 実装後の実行量を1 variant、773 HMM well-runs、model/LightGBM/PF/Beam/booster 0、control再実行0と再確認した。
- Jupytext round-trip、構文、ruff F821、7 synthetic/contract tests、`make validate-exp`、`make validate-template`を通した。
- ユーザーの「実行してください」により、1 scientific variant、773 HMM well-runs、model/LightGBM/fold/PF/Beam/booster/control再実行0のKaggle CPU train実行承認を得た。GPU、internet、inference、submissionは無効のままとした。
- 53文字のfull-directory slugはKaggle SaveKernel 400で拒否され、pull 403により未作成を確認した。意味を保つ`swt`略記の48文字canonical id/titleへそろえ、同じexp内で再packageすることにした。
- 48文字canonical packageのmetadata/bootstrapを再検証し、`kentookumura/exp305-tempered-raw-swt-exact-hmm-emission-train` v1をpushしてKaggle CPU実行を開始した。
- v1は計算前にexp304 silent-fallback field参照ミスでERROR。exp304小型manifestで実値0を確認し、manifest参照へ修正して回帰testを追加した。8 tests、構文、F821、Jupytext、strict validationを通した。
- 修正版packageのNotebook/bootstrapを再照合し、同一kernelへv2としてpushしてKaggle CPU実行を開始した。
- v2が少なくとも2時間25分`RUNNING`を維持した後、ユーザー依頼に従ってKaggle実行は継続したままCodex側の定期監視だけを停止した。
- v2は773/773 HMM完了後、保存cacheを`likpf_mean`絶対値と誤認したlate readoutでERROR。実schemaの`likpf_mean_d`をlast-knownへ加える契約を確認し、v2 outputが空でprediction救出不可と確認した。
- `likpf_mean_d + last_known_tvt`復元と全入力schema fail-fast guardを実装し、9 tests、構文、F821、Jupytext、strict validation、bootstrap照合をPASS。同一kernelへv3をpushし、RUNNINGを一度だけ確認した。
- ユーザーの完了連絡後にv3 `COMPLETE`、runtime 15,983.840秒、3,783,989 rows / 773 wells / 773 HMM runs、prediction content SHA `86b1768f...7302`を確認した。
- directは`11.938287 → 13.218199`、fixed likPF 50/50は`10.269693 → 10.767674`へ悪化し、両方とも改善1/5 folds、1000+/hidden-like 2面/p95/worstを全FAILした。
- strict saved-likPF baseline parityは約`3e-6 ft`差でFAILしたが、科学的悪化`0.498--1.280 ft`より十分小さくnegative decisionは不変と記録した。
- 事前登録どおり救済せずexp305を閉じ、exp304 reserved案3/案4、inference、submissionを無効のまま維持した。
