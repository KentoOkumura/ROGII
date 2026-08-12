# 要件

## 依頼

`KAGGLE_DIRECTION.md` の未着手バックログ
`candidate_perturbation_augmentation_for_likelihood_ranker` を実験化する。
fold-safe な固定 PF / HMM / Beam / dense / geometry 候補を正解非依存に摂動し、
候補の within-10ft likelihood と絶対誤差を学ぶ candidate-long 教師例を増やす。

## 制約

- Route: `ensemble`。exp237 の PF / Beam / HMM / geometry 候補と LightGBM ranker の双方が本質的に寄与する。
- 親: `exp237_hmm_exp226_candidate_selector_on_exp183`。11候補、target-free context、固定Viterbiを維持する。
- exp218 は現行ML比較anchorとして参照するが、初回probeでは候補追加・最終ML再学習をしない。
- augmentation parameter は target、true error、oracle rank、hidden-like role、Public LBを参照せず決定する。
- outer-train wellだけへaugmentationを適用し、outer-validはclean original候補だけで評価する。
- original-only controlとaugmentationを同一outer well GroupKFoldで再学習し、変更点をaugmentationだけにする。
- augmented candidateは教師例専用であり、direct平均、置換、blend、Viterbi state、submission候補には使わない。
- PF/HMM/Beam/exp218 parent/controlを再生成・再学習しない。
- Kaggle CPUを使用し、GPU・internet・inference・submissionは無効にする。
- 再現性: `docs/06_reproducibility.md` に従い、stable hash seed、入力SHA、feature schema SHA、model manifest SHA、OOF prediction content SHAを記録する。

## 受け入れ基準

- `config.yaml` に摂動種別、shift grid、drift、dropout、spread、sampling cap、variant/config/fold/booster数が明示されている。
- `original_only` と `perturbation_augmented` のcandidate-error regressor / within10 classifierを同一5-foldで比較できる。
- augmentation sample生成がstable keyから決定され、同じseed/config/inputでmanifest SHAが一致する。
- candidate値を変える摂動ではcandidate依存のminus-last、候補間spread/rank、multi-observation score/MAE/NCCを再計算する。
- clean outer-valid上でAUC、logloss、Brier、expected-error calibration、topK coverage、selected RMSE、margin calibrationを保存する。
- fixed Viterbi、distance `1000_plus`、exp115 hidden-like、by-well、worst-wellを監査する。
- validation target/error/oracleをfeatureへ流さない自動assertionがある。
- notebookはJupytext percent形式から生成され、入力、augmentation、fold学習、評価、生成物、SHAがセル上で追える。
- gzip生成物はdecompressed content SHAを主証拠として記録する。
- 初回実行前にCPU 2 variants x 2 LightGBM objectives x 5 folds = 20 boosters、parent/control再学習なしを`SESSION_NOTES.md`へ記録する。
