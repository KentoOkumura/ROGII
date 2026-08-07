# exp274_catboost_final_regressor_swap_on_exp238 セッションノート

## 目的

`exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218` の final TVT regressor だけを
LightGBM から CatBoost に差し替え、現行の強い feature surface における model-family
diversity と単体改善を監査する。CatBoost のハイパーパラメータは保存済み公開 notebook
`pixiux/rogii-dual-pipeline-blend` の先頭 config `cb0` を使う。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train version 1 raw guard FAIL、reference-only inference / submission scored完了
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 比較基準: 保存済み exp238 `lgb_mean` OOF 7.936689854
- CV: CatBoost 8.183503603 / parent 7.936690031 / delta +0.246813573
- LB: Public LB 7.715（`ref=54793316`、Kaggle API `COMPLETE`）
- inference: 採用はdisabledのまま。reference-only kernel version 1 `COMPLETE`

## GPU 学習コスト契約

- active variant: 1 (`catboost_public_cb0`)
- CatBoost config: 1
- outer folds: 5
- 合計新規 model: 5
- 1 model あたり最大 iterations / trees: 8,000（early stopping あり）
- 理論上の最大合計 iterations / trees: 40,000
- selector retraining: 0 models
- parent/control LightGBM retraining: 0 models
- 固定比較: 保存済み exp238 final OOF
- Kaggle GPU push: 2026-07-17 に上記コスト契約を提示し、ユーザー承認済み。

## 公開 notebook 設定

- source: `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260611/pixiux__rogii-dual-pipeline-blend/rogii-dual-pipeline-blend.ipynb`
- source SHA256: `9f80687b9582b9b47a464613433afabe74274565252a2e235c152456a0d828e8`
- selected config: `cb0`
- model params: `iterations=8000`, `depth=7`, `learning_rate=0.02`, `l2_leaf_reg=2.0`, `min_data_in_leaf=15`, `border_count=254`, `loss_function=RMSE`, `task_type=GPU`, `od_type=Iter`, `od_wait=300`, `verbose=0`, `random_seed=7`
- fit params: `early_stopping_rounds=250`, `use_best_model=True`
- runtime-only: `devices=0`, `allow_writing_files=false`
- `cb1` は実行しない。

## Kaggle train version 1 結果

- kernel: `kentookumura/exp274-catboost-final-regressor-exp238-train`
- version / id_no: `1` / `127597836`
- status / runtime: `COMPLETE` / T4 / 3,256.205秒
- raw CatBoost: RMSE 8.183503603、MAE 5.036106847、within10 0.864122755
- saved parent: RMSE 7.936690031、MAE 4.929301733、within10 0.866401303
- raw delta: RMSE +0.246813573
- fixed `0.75 * parent + 0.25 * CatBoost`: RMSE 7.950393906、delta +0.013703875
- raw improved folds: 1/5（fold 2だけ -0.017196821）
- worst fold: fold 4、delta +0.702349604
- distance 1000+: delta +0.271066745
- hidden-like spatial / typewell-purged: +0.274254636 / +0.274986218
- worst well `2fd68f7b`: delta +12.293691635
- raw guard: 全項目 FAIL、`all_raw_guards_pass=false`、`inference_allowed=false`
- selector input caveat: source summary は `selector_guard_failed_final_train_forbidden` / `selector_guard_pass=false`。同じ保存済み surface 上の estimator 比較に限定して解釈する。

## Reference inference契約

- authorization: 2026-07-18 ユーザーの「参考のため推論も行ってください」
- purpose: prediction distributionの参考確認のみ。train rejection、anchor、採用判断を変更しない
- primary output: raw CatBoost `submission.csv`
- comparison outputs: 保存済みparent LightGBM再推論、固定`0.75 parent + 0.25 CatBoost`
- saved models: CatBoost 5、parent LightGBM 15、selector LightGBM 20
- training / retraining: 0 models / 0 boosters
- current-test features: exp238 hidden-safe current-test regenerationと同じ380 base + outer-fold matched 35 rank-slot = 415列
- Kaggle runtime: T4、internet off。feature replayと保存済みmodel inferenceのみ
- competition submit: false。output取得とsubmit-checkまでで停止する
- fallback rows: 0を必須とする

## Reference inference version 1 結果

- kernel / version / id_no: `kentookumura/exp274-catboost-final-regressor-exp238-inference` / 1 / `127707471`
- status / runtime / machine: `COMPLETE` / 425.779秒 / T4
- rows / wells / feature: 14,151 / 3 / 415（380 base + 35 selector）
- loaded models: CatBoost 5 / parent LightGBM 15 / selector LightGBM 20、fit 0
- fallback rows: 0
- raw vs parent test prediction RMSE / abs mean / abs max: 1.270216 / 0.966147 / 4.244141 ft
- fixed0.25 blend vs parent test prediction RMSE / abs mean / abs max: 0.317559 / 0.241541 / 1.060547 ft
- submit-check: raw CatBoost / parent / fixed0.25 blendの3件すべてFAIL 0 / WARN 0、公式sampleとID順完全一致
- code submission: root raw CatBoost `ref=54793316`、Public LB 7.715、status `COMPLETE`
- submitted anchor: exp257 7.718を-0.003更新するML route submitted anchor。ensemble anchor exp082 7.601は維持
- adoption: false。train rejectionは変更せず、LB submitted anchor更新とtrain-side採用を分離する

## コマンドログ

- 2026-07-17: `make new-steering EXP=exp274_catboost_final_regressor_swap_on_exp238` で steering docs を作成。
- 2026-07-17: `make new-exp EXP=exp274_catboost_final_regressor_swap_on_exp238 SOURCE=templates/experiment` で実験を作成。
- 2026-07-17: 公開 notebook と exp098 の source extraction 実装の SHA256 を記録。
- 2026-07-17: Jupytext percent 形式の train / inference `.py` を実装。
- 2026-07-17: Jupytext で `.ipynb` へ変換し、`--test`、`py_compile`、full Ruff を実行し PASS。
- 2026-07-17: `.venv/bin/python scripts/validate_experiment.py --experiment exp274_catboost_final_regressor_swap_on_exp238` で strict validation PASS。
- 2026-07-17: canonical id `kentookumura/exp274-catboost-final-regressor-exp238-train` で Kaggle train package を生成。metadata は private / GPU T4 / internet off / run-on-push。
- 2026-07-17: bootstrap ZIP 29 filesをread-only監査し、埋め込み `config.yaml` と loose config の byte-identical、CatBoost config 1、fold 5、model 5、parent retraining false、public source SHA、exp238 engine同梱を確認。
- 2026-07-17: repository test は `119 passed in 22.88s`。
- 2026-07-17: ユーザーの「続きを実行してください」を、提示済み GPU コスト契約（1 variant / 1 config / 5 folds / 5 CatBoost models / parent control retraining なし）での train push 承認として記録。
- 2026-07-17: `kaggle kernels push -p experiments/exp274_catboost_final_regressor_swap_on_exp238/kaggle/train --accelerator NvidiaTeslaT4` を実行し、canonical kernel version 1 の push に成功。
- 2026-07-17: `kaggle kernels pull kentookumura/exp274-catboost-final-regressor-exp238-train -m` で Kaggle 側の存在（`id_no=127597836`）と `enable_gpu=true` / `machine_shape=NvidiaTeslaT4` を確認。
- 2026-07-18: ユーザーから完了連絡を受け、`kaggle kernels status` で `COMPLETE`、通常 logs で `train_completed_raw_guard_failed` を確認。
- 2026-07-18: logs の DataFrame 表を補うため、`kaggle kernels output --file-pattern` で metrics / fold / bucket / hidden-like / by-well / guard / summary / manifest の小さい CSV/JSON だけを `/tmp/exp274-results.flTJwY` に選択取得。model 重みと 378万行 OOF は取得していない。
- 2026-07-18: CatBoost raw 8.183503603（parent +0.246813573）、固定0.25 blend 7.950393906（+0.013703875）、全raw guard FAILを確認し、inference / submit 不採用としてbranchを閉じた。
- 2026-07-18: `metrics.json` JSON parse、strict experiment validation、`review_exp_docs.py` の core evidence reviewをPASS。repository testsは `124 passed in 34.85s`。
- 2026-07-18: reference-only overrideをsteering / configへ追記し、exp238 hidden-safe current-test regenerationを移植。raw CatBoost 5-model平均をprimary、保存済みparent 15-model平均と固定0.25 blendを比較用として実装した。学習処理はなし。
- 2026-07-18: inference Jupytext round-trip、py_compile、full Ruff、strict experiment validationをPASS。親exp238 inference 540行に対しexp274 inferenceは895行 / 8章で、override、CatBoost/parent両model契約、3 prediction、SHA、reference-only境界をnotebook上に展開した。
- 2026-07-18: canonical inference packageを生成。metadataはprivate / T4 / internet off / run-on-push、設定済み10 kernel sourcesとcompetition sourceを確認した。bootstrap ZIP 29 filesのmanifest / size / SHA、埋め込みconfigとloose configのbyte parity、主要7依存sourceを監査してPASS。repository testsは`124 passed in 16.64s`。
- 2026-07-18: `kaggle kernels push -p experiments/exp274_catboost_final_regressor_swap_on_exp238/kaggle/inference --accelerator NvidiaTeslaT4`でcanonical inference kernel version 1をpushした。
- 2026-07-18: ユーザーの完了連絡後、`kaggle kernels status` / logsでkernel version 1 `COMPLETE`、inference summary status `reference_inference_completed_not_submitted_raw_guard_failed`、runtime 425.779秒を確認した。
- 2026-07-18: `kaggle kernels output --file-pattern`でroot submission、parent / fixed blend submission、prediction gzip、feature schema、summaryだけを`/tmp/exp274-inference-output.dRdJvh`へ取得。大きいselector surfaceは取得していない。
- 2026-07-18: 公式`sample_submission.csv`を一時取得し、3 submissionすべてsubmit-check PASS。ID順完全一致、float32 blend式完全一致、submission / schema / decompressed prediction SHAのsummary一致を確認した。
- 2026-07-18: 初回確認ではlatest submission `ref=54793316`をroot raw CatBoostへ帰属したが、Kaggle CLI 2.2.2 / 2.2.3とmonitor one-shotはいずれも`PENDING` / Public Score空欄だったため、API値を推測せず反映待ちとして記録した。
- 2026-07-18: inference / submission記録更新後にmetrics JSON / config YAML parse、strict experiment validation、core evidence reviewをPASS。repository testsは`125 passed in 41.68s`。
- 2026-07-18: ユーザーからPublic LB 7.715の完了連絡を受領。提出一覧先頭は新しい別submission `ref=54798337`だったため、汎用monitorのlatest行をexp274へ誤帰属せず、対象`ref=54793316`をpage内で直接照合して`COMPLETE` / 7.715を確認した。
- 2026-07-18: 7.715はexp257 7.718より-0.003、exp238 hidden-safe 7.775より-0.060、exp218 7.843より-0.128、ensemble anchor exp082 7.601より+0.114。ML submitted anchorだけをexp274へ更新し、CV +0.246814悪化によるtrain rejectionは維持した。
- 2026-07-18: score確定記録後、metrics / configの7.715と`COMPLETE`一致、strict experiment validation、core evidence reviewをPASSした。

### Kaggle 実行コマンド

```bash
make prepare-kaggle-notebooks EXP=exp274_catboost_final_regressor_swap_on_exp238 \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp274-catboost-final-regressor-exp238-train --title 'exp274 catboost final regressor exp238 train' --run-on-push --strict"
make push-kaggle-train EXP=exp274_catboost_final_regressor_swap_on_exp238
kaggle kernels status kentookumura/exp274-catboost-final-regressor-exp238-train
kaggle kernels logs kentookumura/exp274-catboost-final-regressor-exp238-train
make prepare-kaggle-notebooks EXP=exp274_catboost_final_regressor_swap_on_exp238 \
  EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp274-catboost-final-regressor-exp238-inference --title 'exp274 catboost final regressor exp238 inference' --run-on-push --strict"
make push-kaggle-infer EXP=exp274_catboost_final_regressor_swap_on_exp238
```

## 変更点

- exp238 の selector artifact、outer fold role、candidate surface、380 base features、35 rank-slot features、residual target を固定。
- final estimator のみ公開 CatBoost `cb0` に差し替え。
- 各 fold の input matrix content SHA、5 CatBoost model SHA、OOF decompressed SHA、feature schema SHA を保存。
- raw CatBoost と固定 `0.75 * exp238 + 0.25 * CatBoost` の overall / fold / distance / hidden-like / by-well readout を実装。
- raw CatBoost の overall、1000+、hidden-like 2面、worst-well、3/5 folds の guard を実装。
- inference notebook はguard不通過を記録したまま、ユーザー明示承認時だけ動くreference-only境界へ変更した。

## Notebook 構成比較

- 親 exp238 train: 7章、217行。上位 orchestration は notebook にあるが final LightGBM fold 学習は親 helper に委譲。
- exp274 train: 8章、公開設定監査、GPU cost gate、parent OOF、fold matrix、CatBoost fit、stress readout、SHA 保存を notebook セル上に展開。
- 親 exp238 inference: 7章、540行。exp274 inference: 8章、895行。current-test regenerationを維持し、reference override、CatBoost/parent両model契約、3 prediction、SHA、submit禁止を追加。
- 同一 exp 内 helper import は使わず、親 exp238 / exp237 / exp218 の固定 source だけを bootstrap する。

## 再現性メモ

- seed policy: 公開 `cb0` の `random_seed=7`。
- stochastic components: CatBoost GPU training。Python global RNG は使わない。
- CPU/GPU runtime: Kaggle T4 GPU / device 0。CatBoost GPU の bitwise 一致は仮定しない。
- input SHA: parent OOF decompressed `0e7390ac3b3a432b1d432e432cb374cbf38da393a9b95f8f0d6c22732030010c`。selector summary、5 nested score gzip、hidden-like assignment も summary に保存。
- feature SHA: schema `f0c11f34137de7ad011c7a8317ce24c3fafb4e09f9aaf5d064efd8c7ea2494a0`。fold ごとの train / valid float32 matrix content SHA は model manifest に保存。
- model SHA: fold 0-4 は `d1bcfed0...` / `a1530f2e...` / `839b30c3...` / `c126621e...` / `221f7795...`。manifest SHA は `cba180df02928d66698a67970f774278a12f6c536a7e80e23546784e82614028`。
- prediction SHA: OOF decompressed `56a7f1bbeef0e703af74650d41e546343aa6f499a71b584f1a16992a5209aa55`。
- summary SHA: `181f1564014d81bee484a064d54601fc1e727c67a6b7682d9515fb5a87d28939`。
- inference prediction decompressed SHA: `67b4f9fa3d402dd962f495d0218347ba0d677d75d19575b78974ac022395dfaf`。
- reference submission SHA: raw `565c82f8d2a0118fde4741be9f1c510198189e662224db40b03eddc2da074dc5` / parent `829709d6a4a27c7440412ae1b24aeab51734b30b19f59a78e9d0178dadcf6e0e` / fixed blend `a2d64a5f9053a847f2ea365c8b32262992b542254d861dce0fcb8c8f63b5b6aa`。
- inference summary SHA: `3dc6aa51b54666be90c4959d025040600878925e0b052d9e195ac56eac9f6a27`。

## 次のアクション

1. 完了。train-side guard FAILと不採用判断を維持し、追加CatBoost rescueは行わない。
2. route比較ではexp274をML Public-LB submitted anchor、exp082 7.601をensemble anchorとして分離して扱う。
