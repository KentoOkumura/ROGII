# exp281_exp226_residual_offset_exact_hmm_transition_probe

## 状態

- ルート: `pf_beam`
- 状態: Kaggle private CPU version 1完了 / train-side guard FAIL / negative close
- CV: 9.827420
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-19
- shape親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- decoder親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 先行条件: `exp280_exp226_shift_likelihood_separability_readout` guard PASS
- 失敗参照: `exp279_exp226_geop_centered_exact_hmm_redecode`

## 仮説

exp226 `tvt_geop`は局所形状を捉え、exp280はその周囲のtruth-nearest shiftをraw GR likelihoodが
5 foldsでshuffledより良く順位付けできることを示した。absolute TVTを自由探索せず、
`TVT_t = exp226_geop_t + delta_t`としてslow offset `delta_t`だけをexact HMMで時系列統合すれば、
exp279の固定absolute unaryより安全にpersistent offsetを減らせる可能性がある。

## 変更点

- exp226 group-safe OOFの`tvt_geop`をmoving coordinate centerとして固定する。
- HMM position stateをabsolute TVTから`delta`へ置き換え、補助状態をoffset-rateへ限定する。
- absolute transition centerを`diff(tvt_geop)`、delta transitionを`offset_rate * dMD`とする。
- offset grid `[-80,80] ft`、step `0.35 ft`、41 rate states / span `+-0.10`を事前固定する。
- exp209 Gaussian raw-GR emission、known-prefix sigma、missing-GR、process grammarを固定する。
- exp226`tvt_pred` / `gr_delta` / truth / errorをdecoder inputから除外する。
- 全well path生成後にだけtruthと保存済みcontrolを別読込して結合する。
- inference notebookはtrain guard通過とユーザー承認までfail-closedにする。

## 検証方針

- Fold: exp226保存済みgroup-safe 5 folds
- Group: `well`
- Score rows: unknown suffix 3,783,989 rows / 773 wells
- Promotion baseline: exp263 fixed OOF 8.238331
- Guard: overall gain 0.02以上、改善3/5 folds、near / 1000+ / hidden-like悪化0.02以下、worst-well +0.25以下
- Recovery guard: persistent-offset episode数はexp263以下、256/512行復帰率はexp263以上
- Technical guard: exp263 parity `1e-5 ft`、delta-grid / finite coverage 100%、入力decompressed SHA一致
- 実行量: HMM 1 variant / 773 well-runs / LightGBM config 0 / trained fold 0 / booster 0

## 実行入口

- 学習 notebook: `exp281_exp226_residual_offset_exact_hmm_transition_probe_train.ipynb`
- 推論 notebook: `exp281_exp226_residual_offset_exact_hmm_transition_probe_inference.ipynb`（fail-closed）
- Kaggle準備: `make prepare-kaggle-notebooks EXP=exp281_exp226_residual_offset_exact_hmm_transition_probe ...`
- canonical kernel候補: `kentookumura/exp281-exp226-residual-offset-exact-hmm-train`
- notebook実行: Kaggle CPUを正とし、ローカルfull notebookは実行しない。

## 結果

| メトリック | 値 |
| --- | --- |
| 実装検証 | 6 tests PASS、構文 / ruff / Jupytext / strict exp validation PASS |
| Kaggle package | train package生成済み、CPU / offline metadataとsource/config SHA一致 |
| residual-offset HMM | 9.827420 |
| exp263 fixed | 8.238332 |
| delta | +1.589088 ft |
| 改善fold | 0/5 |
| persistent episodes | 530 vs 551（PASS） |
| worst-well delta | +30.961675 ft（FAIL） |
| Public LB | - |
| Private LB | - |

## 生成物

- OOF候補: `artifacts/exp281_exp226_residual_offset_exact_hmm_transition_probe_oof_predictions.csv.gz`
- 評価表: candidate / fold / distance / hidden-like / by-well metrics
- 復帰診断: persistent-offset episodes / recovery summary
- 再現性証拠: input / well / decoder manifest、summary、raw・decompressed・logical SHA
- 実行前package: `kaggle/train/`（Kaggle output artifactはCPU実行後に生成）

## 所見

### 良かった点

- exp280のpositive diagnosticをhard correctionへせず、slow-offset grammarだけへ接続した。
- score-stageとtruth/control attachmentを全well path freeze境界で分離した。
- parameter rescueを防ぐ1 fixed scientific contractをconfigとtestで固定した。
- 773 wells中408 wells、MAE、within5、persistent episode数、256/512 recoveryは改善した。

### リスク / 注意

- 全5 folds、near / 1000+ / hidden-like 2面が悪化し、rare tailがRMSEを支配した。
- exp280の局所shift識別力はglobal always-on offset decoderの安全性には不足した。
- 1成功runだけではdeterministic anchorと呼ばない。
- guard FAIL後のgrid/process/rate/likelihood探索、PF/blend/selectorは禁止する。

## 次

- branchを閉じ、独立したprefix-masked offset readoutとfuture-evidence回復監査を優先する。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせる。
