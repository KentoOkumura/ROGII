# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先0-booster backlog
`exp226_shift_likelihood_separability_readout` を、未使用番号
`exp280_exp226_shift_likelihood_separability_readout` として実装する。

group-safe exp226 OOFのgeometry-only `tvt_geop`を局所形状として固定し、事前固定した
対称shift bankだけを加える。raw horizontal GRとtypewell GRから作るtarget-free likelihoodが、
true TVTに最も近いshiftを512行block単位で順位付けできるかを監査する。

## 制約

- Route: `pf_beam`。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- shift bankは`[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`の13候補に固定する。
- blockはwellのunknown suffix先頭から非重複512行に分割し、末尾short blockも保持する。
- likelihoodはexp209/exp279と同じGaussian raw-GR emissionを使う。known prefix residual std、
  `sigma clip=[10,60]`、missing-GR interpolation、log-likelihood clip 600を固定する。
- candidate tieはconfig上のshift順、truth-nearest tieも同じ順で解く。
- stable shuffled-score controlは`SHA256(experiment, seed, well, block)`由来のlocal RNGだけを使う。
- exp226 `tvt_pred` / `gr_delta` / `tvt_true` / `error` / `abs_error`をtarget-free score APIへ渡さない。
- true TVTは全wellのscore tableを凍結してcontent SHAを確定した後にだけ結合する。
- 1 audit variant、LightGBM config / trained fold / booster / HMM decodeは`0 / 0 / 0 / 0`。
- direct shift correction、candidate平均、selector、raw-test inference、submissionを行わない。
- shift幅、grid、block、calibration、missing-GR処理、score集約、guardを同一OOF結果で変更しない。

## 受け入れ基準

- exp226 canonical OOFのdecompressed SHA、3,783,989 rows、773 wells、fold 0-4をhard guardする。
- raw/typewell入力とOOF row identityを重複・欠損なしで整合し、全target-free candidate scoreがfiniteである。
- target-free score long table、truth-attached block readout、fold/scope/shift/by-well metrics、
  persistent-offset episode、well/input manifest、summaryを保存する。
- top1、top3、MRR、offset符号一致をrealとstable shuffled controlで同じ実装から計算する。
- separability guardは4指標のrealがshuffledを5/5 foldsすべてで上回る場合だけPASSとする。
- near、1000+、hidden-like spatial/typewell-purged、persistent-offset blockを独立scopeとして読む。
- inference notebookはfail-closedとし、`submission.csv`を生成しない。
- Jupytext percent source、ipynb、py_compile、Ruff、unit test、strict experiment validationを通す。
- 本実験はprediction/submission anchorではなく固定入力へのdeterministic diagnosticとして扱う。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
