# exp280_exp226_shift_likelihood_separability_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU train-side readout完了、固定guard PASS
- CV: -（予測CVではなくdiagnostic）
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-19
- 科学的親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 失敗参照: `exp279_exp226_geop_centered_exact_hmm_redecode`

## 仮説

exp226 `tvt_geop`は局所形状を捉えており、主な失敗が低周波の縦offsetなら、形状を固定した
13本のshift候補のうちtrue TVTに近い候補をraw GR/typewell likelihoodだけでfold-stableに
順位付けできる。これを先に確認すれば、後続のresidual-offset HMMへ進む価値を分離できる。

## 変更点

- exp226 group-safe OOFの`tvt_geop`だけを固定入力にする。`tvt_pred` / `gr_delta`は禁止する。
- shift bankを`[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`へ固定する。
- unknown suffixを先頭から非重複512行に分け、末尾short blockも保持する。
- exp209 Gaussian raw-GR emissionを固定し、各shiftのblock mean log-likelihoodを作る。
- 全target-free scoreを凍結・SHA化した後にだけtrue TVTを結合する。
- top1/top3/MRR/符号一致、margin、regret、bank coverageをreal/stable shuffledで比較する。
- LightGBM / HMM / PF / correction / inference / submissionは実行しない。

## 検証方針

- Fold: exp226保存済み5 foldsをreadout strataとして使用。trained foldは0。
- Group: `well_id`。
- Score rows: train unknown suffix 3,783,989 rows / 773 wellsを512行blockへ集約。
- Scope: fold、near、1000+、hidden-like 2面、persistent-offset block。
- Leakage check: safe列だけのscore-stage読込、score table content SHA確定後のtruth再読込、
  score APIでtruth/error列をhard reject。
- Guard: top1/top3/MRR/signの4指標がstable shuffledを5/5 foldsで上回ること。
- 実行量: audit variant 1 / LightGBM config 0 / trained fold 0 / booster 0 / HMM 0。

## 実行入口

- 学習 notebook: `exp280_exp226_shift_likelihood_separability_readout_train.ipynb`
- 推論 notebook: `exp280_exp226_shift_likelihood_separability_readout_inference.ipynb`（fail-closed）
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp280_exp226_shift_likelihood_separability_readout`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| guard | PASS（top1/top3/MRR/signが各5/5 foldsでshuffled超え） |
| top1 | 0.189547（shuffled 0.075767、lift +0.113779） |
| top3 | 0.452421（shuffled 0.234493、lift +0.217927） |
| MRR | 0.389626（shuffled 0.245536、lift +0.144090） |
| sign | 0.498523（shuffled 0.418518、lift +0.080005） |
| coverage | row identity 1.0 / finite score 1.0 / bank range 1.0 / quantization 1.0 |
| runtime | 456.972秒、7,787 blocks / 773 wells / 3,783,989 rows |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- approved contractを1つのconfigへ固定し、同一OOFでのshift/grid/calibration救済を禁止した。
- score-stageとtruth readoutをAPI・読込列・content SHAで分離した。
- 合成データで+10 ftのtruth-nearest shiftがlikelihood top1になることをunit testした。
- 4 separability指標が全5 foldsでstable shuffledを上回り、識別力のfold安定性を確認した。
- 1000+、hidden-like 2面、persistent-offsetでも4指標のliftは正だった。

### 悪かった点

- top1は18.95%、persistent-offsetでも15.05%に留まり、hardな直接shift選択には弱い。
- signの絶対精度は49.85%で、shuffled比では改善したが単独補正ruleには使えない。
- near scopeは1 block / 1 wellだけで、独立したnear性能の根拠にならない。

### リスク / 注意

- typewell端ではexp209互換のendpoint holdを使うため、native/extended coverageを別保存する。
- guard PASSでも補正採用ではなく、別実験のresidual-offset HMM検討を許可するだけ。
- guard FAIL後にshift幅やlikelihood calibrationを調整しない。
- target-free score content SHAは`4a546cfe5f9291168bdb4dcb912182b079e0343af845f76005f6a7100ac3aa46`。
- Kaggle kernelはversion 1 / id_no `127828902`、inference / submissionは未実行。

## 次

固定guard通過を先行条件としていた`exp226_residual_offset_exact_hmm_transition_probe`を、
別実験として設計する。exp280のtop1を直接補正として使わず、exp226座標系のslow offset stateへ
raw-GR likelihoodを時系列統合する1 fixed grammarだけを検証する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
