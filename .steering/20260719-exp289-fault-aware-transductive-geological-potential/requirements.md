# 要件

## 依頼

物理モデル単独で Public LB 6.5 を狙う抜本案として、train の formation surface と
hidden well の既知 `TVT_input` prefix を使い、断層を許した共通 2D 地質ポテンシャルを
全 hidden-like well 同時に推定する実験を設計する。初回は backlog、steering、実験ディレクトリ、
設定・評価契約までを確定した。2026-07-19の追加依頼`exp289を実装してください`により、
Stage 0 notebook/source/testsとdisabled inferenceまでを実装対象へ追加する。Stage 1/2、Kaggle実行、
raw-test inference、提出は引き続き対象外とする。

## 制約

- Route: `pf_beam`。LightGBM/CatBoost/XGBoost/NN、既存 ML anchor、candidate selector、blendを予測生成に使わない。
- 単一モデル: 出力は一つの fault-aware geological-potential MAP 解だけとし、候補曲線bank、best-of-N、hard/soft selector、posthoc well補正を作らない。
- 物理式を `F_w(r) = Z_w(r) + TVT_w(r) + c_w` とし、予測は最後の既知row `a`から
  `TVT_hat(r) = TVT_input(a) + F_hat(r) - F_hat(a) - (Z(r) - Z(a))` で一意に生成する。
- canonical formationは`ANCC`とする。他の`ASTNU/ASTNL/EGFDU/EGFDL/BUDA`はStage 0の平行性・断層coherence監査に限り、初回Stage 1 solverの追加観測にはしない。
- outer-valid wellのformation 6列と予測対象true `TVT`はgraph/scale/fault/threshold/solver fitから完全に除外する。
- outer-validでは`MD/X/Y/Z/GR/TVT_input`だけをhidden相当入力とし、fold内wellを一括transductive推定する。
- Stage 1はGRを使用しない。GR因子はStage 1の事前guard通過後、同じexp内のStage 2として別承認を得るまで未実装とする。
- exp226、exp263、HMM/PF/Beam出力は比較用保存OOFとしてのみ読み、モデル入力、初期値、補正、blendには使わない。
- oracleは評価指標に使わない。row/segment/well oracle、best-of-N、truth-nearest shift、oracle prediction保存を禁止する。
- 初回実装は一つの固定scientific contractだけとし、mesh、neighbor数、robust cutoff、smoothness、GR weightのgrid探索を行わない。
- 再現性: `docs/06_reproducibility.md` に従い、graph、fold、入力、fault-edge weight、OOF、predictionのschema/content SHAを記録する。
- Kaggle Notebook実行を正とし、ローカル実行はユーザーが明示承認したsmoke debug以外では行わない。
- 実装前に、active variant 1、ML config 0、trained fold 0、booster 0、control再学習0であることを`SESSION_NOTES.md`へ再確認する。

## 受け入れ基準

### 今回の設計確定

- `.steering/20260719-exp289-fault-aware-transductive-geological-potential/`に要件、設計、タスクリストがある。
- `experiments/exp289_fault_aware_transductive_geological_potential/`にplanned状態のscaffold、`config.yaml`、日本語の記録がある。
- `KAGGLE_DIRECTION.md`未着手バックログの最上位高リスク枠に、先行条件、段階guard、禁止事項を記録する。
- `experiment_summary.md`にplanned実験として記録する。
- notebook/source実装、Kaggle package prepare/push/run、inference、submissionを行っていない。

### Stage 0: 0-booster fault仮説監査

- 5 outer foldsすべてでouter-validのformation/true suffixをgraph構築前に削除し、forbidden-column hit 0をmanifestで確認する。
- formation 6面について、within-wellの`delta formation`と`delta(Z+TVT)`のidentity誤差を記録する。
- outer-train formationとouter-valid prefixだけからtarget-free fault-crossing riskを凍結し、その後にexp226 by-well誤差を結合する。
- `abs(exp226 bias) >= 10 ft`識別AUC `>= 0.65`、pooled Spearman `>= 0.25`、fold正方向 `>= 4/5`をすべて満たす場合だけStage 1実装へ進む。
- Stage 0不通過時はthreshold、edge feature、formation面、risk集約の救済gridを行わずbranchを閉じる。

### Stage 0実装

- Jupytext percent形式のcompact self-contained train sourceと読める正規notebookがある。
- outer-valid safe loader、truth-after-freeze、stable edge order、formation identityの専用testsがある。
- disabled inferenceはStage 1別承認前にsubmissionを作らずfail-closedになる。
- active audit variant 1、ML config / trained fold / booster `0/0/0`、control再生成0をconfigと記録で固定する。
- Kaggle package prepare/push/runは別承認まで行わない。

### Stage 1: GRなし単一MAP surface

- active variantは`fault_aware_transductive_map_no_gr`の1本だけ。保存済みexp226 OOF RMSE `9.4271095966`を比較基準とし、controlを再生成しない。
- 予測対象全rowに一つのdirect OOF予測を生成し、coverage 1.0、fallback 0、finite 1.0を満たす。
- Stage 2検討の事前guardはdirect OOF RMSE `<= 8.0`、exp226比改善 `>= 4/5 folds`、well RMSE p95 `<= 15.0 ft`、exp226 worst 66 wellsと同じ固定集合のMSE share `<= 45%`とする。
- 物理モデル単独のinference候補化guardはdirect OOF RMSE `<= 7.0`、exp226比 `5/5 folds`改善、hidden-like spatial/typewell-purged両面改善、well RMSE p95 `<= 13.0 ft`、worst-well RMSE `<= 40.0 ft`とする。
- guard未通過ではraw-test inference、submission、GR追加、blend、selector、parameter rescueを行わない。

### Stage 2: 条件付きGR観測因子

- Stage 1事前guard通過後、別承認を得た場合だけ実装する。
- known prefixだけでamplitude/blur/noiseを校正したordered multi-scale Type Well GR event likelihoodを同じMAP objectiveへ一つの弱い因子として追加する。
- GR factorあり/なしを候補bankとして選択せず、事前固定した一つのjoint MAPをdirect OOF評価する。
- deterministic anchorとして扱う場合は、graph/feature content SHA、solver manifest SHA、OOF/prediction SHA、submission SHA、Kaggle kernel versionを記録する。
- gzip生成物を比較する場合はraw `.csv.gz` SHAではなくdecompressed content SHAを主証拠として記録する。
