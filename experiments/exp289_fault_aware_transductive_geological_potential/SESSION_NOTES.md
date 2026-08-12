# exp289_fault_aware_transductive_geological_potential セッションノート

## 目的

物理モデル単独でPublic LB 6.5を狙う新しいモデル系列として、断層を許した共通2D地質ポテンシャルを
全hidden-like well同時に推定するscientific contractを検証する。2026-07-19の追加依頼では、
前段のStage 0 fault-topology association readoutだけを実装し、Stage 1/2は未実装のまま維持する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0科学guard不通過・branch closed
- CV: まだなし
- LB: まだなし
- Kaggle package / run: v1 error / v2 error / v3 complete
- inference / submission: 未承認・未実施

## 2026-07-19 Stage 0 Kaggle CPU実行承認

- 承認: user message `実行してください`。
- active audit variant: 1。
- ML config / trained fold / booster: `0 / 0 / 0`。
- control / parent再生成: 0。保存済みexp226 OOFはfold identityとpost-freeze bias readoutにだけ使用する。
- runtime: Kaggle CPU / float64 / single process / BLAS thread 1 / GPU off / internet off。
- execution scope: Stage 0だけ。Stage 1/2、inference、submissionは実行しない。
- canonical kernel: `kentookumura/exp289-fault-aware-geopotential-stage0-train`。
- Kaggle source: `kentookumura/exp226-k16-kappa-repro-train`。

### Kaggle v1 push

- push成功: `2026-07-19 20:18:22 JST`。
- kernel: `kentookumura/exp289-fault-aware-geopotential-stage0-train` version 1、id_no `127879234`。
- URL: <https://www.kaggle.com/code/kentookumura/exp289-fault-aware-geopotential-stage0-train>
- push直後status: `KernelWorkerStatus.RUNNING`。logsはまだ空。
- private / CPU / GPU off / internet off、competition sourceとexp226 kernel sourceをpull済みmetadataで再確認した。
- pushed bootstrap `config.yaml` SHAは`2154500961bf042a611855044ba40d604f3f643d9ab502f22299956bb46d7f4a`。以降のlocal status更新は実行管理用で、v1内のbootstrap内容はこのSHAで固定する。

### v1入力契約エラーとv2技術修正

- v1 status: `KernelWorkerStatus.ERROR`。fold 0のsource node構築前に`1b1eba53`の全行非有限`ANCC`を検出して停止した。
- solver / model / booster fit: `0 / 0 / 0`。target-free riskも未生成。
- local raw train全773 wellsを監査し、`ANCC`非有限は部分欠損ではなく全行欠損7 wells、45,634 rowsだった。
- 対象well: `03a935ae`, `1b1eba53`, `4c2208f5`, `727a3a10`, `81bf5923`, `a8ed028a`, `d7eb0be8`。
- v2 policy: fold source側で全行非有限`ANCC` wellだけをdonorから除外する。部分欠損はfail-closedを維持する。outer-valid readerは従来どおり`MD/X/Y/Z/TVT_input`だけを読む。
- graph manifestにfold別のassigned source数、finite-formation source数、除外数、除外ID、除外ID SHAを保存する。
- これは入力欠損への技術修正で、Stage 0 risk式、guard、variant数、fold、ML config、booster数は変更しない。
- v2 validation: 専用pytest `9 passed`、全体pytest `261 passed`、`py_compile`、ruff、Jupytext `--test`、strict experiment validationを通過した。
- v2 bootstrap `config.yaml` SHA: `1e0aa19992ad83d15a215aeb57c754323f5f92e4d822be328d6632e3257715f0`、local/loose/bootstrap/manifest一致。
- v2 bootstrap train source SHA: `a444a9aebe48fa7cc9fc7beb7373bf7f3cc2f9b86a8c3f70050c36c0448b92d2`、local/loose/bootstrap/manifest一致。
- v2 bootstrap `project.yml` SHA: `7c933e0240d1b4a8f66a4085bcb9b1eb97d9100f1ac494c8bb789219d39fe157`、local/loose/bootstrap/manifest一致。
- v2 push成功: `2026-07-19 20:26:10 JST`、同じcanonical kernelのversion 2。push直後statusは`KernelWorkerStatus.RUNNING`。
- pushed v2 config/sourceは上記SHAで固定し、push後のlocal status更新は実行管理用として扱う。

### v2 DataFrame attrs errorとv3技術修正

- v2 status: `KernelWorkerStatus.ERROR`。v1の全行欠損`ANCC`処理は通過し、fold 0 source nodesを構築した。
- target nodesを複数wellで`pd.concat`すると、pandasが各frameの`attrs["original_rows"]`を比較し、異なる配列長のbroadcast errorで停止した。
- solver / target-free risk / booster fit: `0 / 0 / 0`。
- v3 fix: `row_idx`をmaterializeした直後にper-well bookkeeping attrsをclearしてからconcatする。target-safe列、sampling、graph、risk、guardは変更しない。
- 異なる行数の2 target wellsを同時に`build_target_nodes`へ渡す回帰testを既存target-safe testへ追加する。
- v3 validation: 専用pytest `9 passed`、全体pytest `261 passed`、`py_compile`、ruff、Jupytext `--test`、strict experiment validationを通過した。
- v3 bootstrap `config.yaml` SHA: `40a8cdcab3a9a29307c60ba5cdbe4079232061c5cbe84a16c1d8c03bcb0b9899`、local/loose/bootstrap/manifest一致。
- v3 bootstrap train source SHA: `20cdaf42259a77ef40d615c4a3fa5f12d2bdf763d1851a64741410442342bfd4`、local/loose/bootstrap/manifest一致。
- v3 bootstrap `project.yml` SHA: `7c933e0240d1b4a8f66a4085bcb9b1eb97d9100f1ac494c8bb789219d39fe157`、local/loose/bootstrap/manifest一致。
- v3 push成功: `2026-07-19 20:30:33 JST`、同じcanonical kernelのversion 3。push直後statusは`KernelWorkerStatus.RUNNING`。
- pushed v3 config/sourceは上記SHAで固定し、push後のlocal status更新は実行管理用として扱う。

### v3完了・Stage 0判定

- 完了時刻: `2026-07-19T11:34:38.400908+00:00`（`2026-07-19 20:34:38 JST`）。
- Kaggle status: `KernelWorkerStatus.COMPLETE`。runtime `241.548128`秒、peak RSS `693.191406 MB`。
- technical guard: expected folds/wells、forbidden hit 0、source/target overlap 0、source formation accounting、risk finite coverage、truth-before-freeze 0をすべてPASS。
- scientific guard: `abs(exp226 bias)>=10` AUC `0.570651817 < 0.65` FAIL、pooled Spearman `0.127885011 < 0.25` FAIL、positive folds `5/5 >= 4/5` PASS。総合FAIL。
- fold別AUC: `0.515410 / 0.538095 / 0.626333 / 0.562908 / 0.586947`。
- fold別Spearman: `0.079190 / 0.136424 / 0.108868 / 0.175217 / 0.128489`。
- next action: `close_branch_without_rescue_grid`。Stage 1/2、inference、submissionは未実装・未実施のまま閉じる。

### output / SHA検証

- 必要なv3 outputだけを`/tmp/kaggle-output/exp289_fault_aware_transductive_geological_potential/train_v3`へ取得した。
- input manifest 774件、graph manifest 5 folds、target-free node risk 320,991行、well risk 773行、exp226 bias readout 773行を確認した。
- source ANCC全行欠損除外数はfold別`6 / 4 / 6 / 6 / 6`、延べ28。assigned source数とのaccountingは全fold一致。
- formation 6面identityはoverall RMSE最大`0.007182281`、最大絶対誤差`0.030000000`、相関最小`0.997633716`。
- graph manifest frozen SHA: `7040f60dc907bbb5b8c6bb86a05448a9d087445a598c1e24e8477da191d155e0`。
- node risk frozen SHA: `2f2a6320f83237d5b55f5a11dedb9c4adbf4c5ef093b1829be58aafd66cd85af`。
- well risk frozen SHA: `2f0cef466c6ba469573f969d190af5c8bda9b509be68084958ec1d9c67ff061e`。
- summary SHA: `6dcaac7dc05bfb33d4f899720db7209ab19d883a2da413a4b917989efa7925f8`。
- contract SHA: `33adb1f5ce430fe5ab71deaaea68d164c24c85d1da2f1d4ef83760c2e59d53d1`。
- manifest 7生成物のraw SHAとcanonical CSV content SHA、pushed config/source SHAを取得outputで照合しPASS。`submission.csv`なし。

### package prepare / parity

```bash
make prepare-kaggle-notebooks EXP=exp289_fault_aware_transductive_geological_potential \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp289-fault-aware-geopotential-stage0-train --title 'exp289 fault aware geopotential stage0 train' --run-on-push --strict"
```

- metadata: private / CPU / GPU off / internet off / run_on_push true。
- competition source: `rogii-wellbore-geology-prediction`。
- kernel source: `kentookumura/exp226-k16-kappa-repro-train` 1件。
- bootstrap `config.yaml` SHA: `2154500961bf042a611855044ba40d604f3f643d9ab502f22299956bb46d7f4a`、短縮後canonical名で再prepareし、local/loose/bootstrap一致。
- bootstrap train source SHA: `4396404f5912e46be2116ddd4aa3e59c285c5dc015b76590655636131b8d7ea0`、local/loose/bootstrap一致。
- bootstrap `project.yml` SHA: `7c933e0240d1b4a8f66a4085bcb9b1eb97d9100f1ac494c8bb789219d39fe157`、local/bootstrap一致。

### 初回push失敗とcanonical名の短縮

- 初回の58文字kernel id/titleではKaggle `SaveKernel` がHTTP 400を返し、versionは作成されずbooster fitも0だった。
- 同idへのmetadata pullはHTTP 403で、kernelが作成されていないことを確認した。
- Kaggle CLIのローカル検証に上限チェックがなく、API応答にも詳細がないため、文字数制約が原因という判断は推定である。
- 入力元`kentookumura/exp226-k16-kappa-repro-train`のmetadata pullは成功し、参照権限を確認した。
- 科学的契約を変えず、同じexp289のcanonical名を44文字の`kentookumura/exp289-fault-aware-geopotential-stage0-train`へ短縮して再prepare / 再pushする。

## 2026-07-19 Stage 0実装

### 実装内容

- Jupytext percent形式の`*_compact_selfcontained_train.py`を正として、9章・1,350行のStage 0 notebookを実装した。
- outer-trainは`MD/X/Y/ANCC`、outer-validは`MD/X/Y/Z/TVT_input`だけをfold別に読む。valid formation 6列とtrue TVTはrisk freeze前にmaterializeしない。
- raw row stride 16、anchor、final row、固定turning pointをnode化し、outer-train median/MAD標準化XY上のcross-well k=12 graphを作る。
- donor ANCC weighted median、weighted MAD、trajectory jump、known-prefix datum residualをfold外scaleでrisk化し、事前固定primary `suffix_fault_risk_p90`をwell単位でSHA freezeする。
- freeze後にだけ保存済みexp226 `error`を読み、`abs(bias)>=10` AUC、risk対`abs(bias)` Spearman、fold方向を計算する。
- formation 6面の`delta formation - delta(Z+TVT)` identity auditもfreeze後だけ実行する。
- inference notebookはsample submissionを作らず、Stage 1別承認前はfail-closedに置き換えた。

### 検証コマンド

```bash
.venv/bin/python -m py_compile experiments/exp289_fault_aware_transductive_geological_potential/*compact_selfcontained*.py
.venv/bin/ruff check experiments/exp289_fault_aware_transductive_geological_potential/*compact_selfcontained*.py experiments/exp289_fault_aware_transductive_geological_potential/tests/test_exp289_fault_aware_transductive_geological_potential.py
.venv/bin/pytest -q experiments/exp289_fault_aware_transductive_geological_potential/tests/test_exp289_fault_aware_transductive_geological_potential.py
# 8 passed
.venv/bin/pytest -q
# 240 passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp289_fault_aware_transductive_geological_potential/*compact_selfcontained*.py
make validate-exp EXP=exp289_fault_aware_transductive_geological_potential
# experiment validation passed (strict)
```

親実験はないためcompact notebookの章立て比較対象はない。設計根拠のexp226 notebookは旧helper型であり、構成移植元には使っていない。

## 2026-07-19 設計確定

### コマンドログ

```bash
task new-steering EXP=exp289_fault_aware_transductive_geological_potential
# task command unavailable
make new-steering EXP=exp289_fault_aware_transductive_geological_potential
make new-exp EXP=exp289_fault_aware_transductive_geological_potential
make update-summary
make validate-exp EXP=exp289_fault_aware_transductive_geological_potential
# experiment validation passed (strict)
```

### 根拠

- exp226保存OOF: RMSE 9.4271095966、well RMSE p95 17.1052204007、worst 58.8026747655。
- exp226 worst RMSE上位66 wellsは全OOF MSEの52.5422%を占める。
- wellごとの平均bias成分をMSE分解したwithin-well centered RMSEは5.7775908563。ただし達成可能oracleではなく故障診断に限る。
- exp138の通常ANCC KNN/local planeはlongtail delta RMSE 25.81/28.25で、smooth spatial interpolationだけでは不足した。
- exp280 raw GR shift top1は18.95%、exp282 self-GR loop closureは直接transferで悪化、exp285 prefix offsetとfull suffix offsetのSpearmanは-0.0041だった。

### 固定した判断

- 新規standalone physics familyとし、exp226のparameter rescueにはしない。
- `F = Z + TVT + c_w`とanchor-relative prediction式を物理モデルの正とする。
- canonical formationはANCC。初回Stage 1はGRなし、一つのjoint MAPだけを出力する。
- fault edgeはouter-train residual MADで正規化したtruncated-quadratic lossとcut-weight TVで扱う。
- 5-fold outer-valid wellsをfoldごとに全件同時推定する。
- Stage 0でfault仮説を反証可能にし、通過しなければStage 1を実装しない。
- oracle、candidate bank、selector、blend、posthoc well補正を禁止する。

## 実行コスト契約

### 今回の実装

- active audit variant: 1
- LightGBM/CatBoost/XGBoost config: 0
- trained fold: 0
- booster: 0
- control再学習: 0
- Kaggle実行: 0

### 将来のStage 0実装時

- active audit variant: 1
- ML config / trained fold / booster: 0 / 0 / 0
- control再生成: 0。保存済みexp226 by-well/OOFを比較にのみ使用する。
- CPU only、GPU/internet off。
- 実装承認: 2026-07-19 user message `exp289を実装してください`。
- Kaggle pushは2026-07-19 user message `実行してください`で承認済み。

## 再現性メモ

- seed policy: canonical pathはRNGなし、fold/well/row/edgeをstable sort。
- stochastic components: なし。
- CPU/GPU runtime: CPU float64、single process、BLAS thread 1、GPU off、internet offを予定。
- Kaggle kernel id / version: 未作成。
- input / fold / graph schema SHA: 実装後にfold別記録。現時点ではなし。
- graph/fault/surface content SHA: 実装後に記録。現時点ではなし。
- solver manifest SHA: 実装後に記録。現時点ではなし。
- prediction / submission SHA: Stage 0では生成しない。Stage 1/推論は未承認。
- deterministic anchor: false。rerunとhidden regeneration parity未確認。
- Kaggle bootstrap: 未prepare。将来prepare後にembedded config/source parityを確認する。

## 実装状態

- Stage 0 train source/notebook、disabled inference source/notebook、専用testsを実装した。
- Stage 1 sparse MAP solver、Stage 2 GR factor、Kaggle packageは未実装・未作成。
- Stage 0の初回実行はKaggle CPUを正とし、ローカルfull runは行っていない。

## 次のアクション

1. Kaggle CPU pushを希望する場合、1 variant / 0 ML config / 0 trained fold / 0 booster / control再生成0を再確認して別途承認を得る。
2. Stage 0のtechnical/scientific guard通過後、Stage 1実装の別承認を得る。
3. Stage 0 FAIL時はthreshold、formation面、risk aggregationを救済せずbranchを閉じる。
4. Stage 1 guard通過後、Stage 2 GR factorまたはraw-test inferenceの別承認を得る。
