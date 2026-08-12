# exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline セッションノート

## 目的

KAGGLE_DIRECTION backlog `discussion711308_dz_dtvt_bpeak_cluster_baseline` を実装する。Kaggle discussion 711308 の `dTVT/dMD ~= a*dZ/dMD + b` と `b` peak cluster による no-ML direct baseline を再現し、Public LB 約 12.8 の再現性と offset / level diagnostic としての価値を切り分ける。

## 2026-07-06 実装

- `.steering/20260706-exp206-discussion711308-dz-dtvt-bpeak-cluster-baseline/` を作成し、requirements / design / tasklist を記入。
- `experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/` を template から作成。
- Route: `pf_beam`
- GPU 学習: なし
- active rule variant 数: 6 (`global_median`, `prefix_fit`, `peak_cluster_median`, `nearest_xy_k8`, `hybrid_peak_xy_k8`, `exact_typewell_peak_xy_k8`)
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- selected inference variant: `exact_typewell_peak_xy_k8`

## 実装内容

- `dz_dtvt_bpeak_cluster_baseline.py`
  - raw train/test horizontal/typewell files から well metadata、prefix fit、full train fit を生成する。
  - consecutive step で `dTVT/dMD ~= a*dZ/dMD + b` を最小二乗 fit する。
  - train full-fit `b` の histogram peak centers を作り、full/prefix `b` peak label を割り当てる。
  - train pseudo-tail では target well を source pool から除外し、prefix-only `b` label、exact typewell hash、XY nearest で local `a,b` を割り当てる。
  - last known TVT から `dTVT = a*dZ + b*dMD` を累積し、candidate 別 prediction / metrics / assignment artifact を保存する。
  - inference では全 train source fit から test prediction と `submission.csv` を生成する。
- `exp206_*_train.py`
  - Jupytext 起点の train notebook source。設定、コスト、入力 contract、train audit、metrics/artifact summary をセル分割。
- `exp206_*_inference.py`
  - Jupytext 起点の inference notebook source。train/test/sample contract、submission 生成、summary 表示をセル分割。

## リークガード

- full train true TVT で得た target well 自身の `full_b` / `full_b_peak_label` は validation assignment に使わない。
- validation target well は source pool から除外する。
- query assignment は `TVT_input` known prefix、MD/X/Y/Z、typewell exact hash だけを使う。
- test inference は test true TVT を持たず、同じ query assignment 関数を使う。

## 再現性メモ

- seed policy: no RNG / deterministic sorting
- stochastic components: なし
- CPU/GPU runtime: Kaggle CPU、GPU disabled
- Kaggle kernel id / version: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v1
- input / feature schema SHA: model なし。train OOF prediction content SHA を feature evidence として記録。
- feature content SHA: `5940a36cb3682760ed6e7cebfc2545034695314ae0f0236454e659451548e5e6`
- model manifest / model SHA: model なし
- prediction SHA: raw gzip `6b4714927fd37042dbd942cf61da09a72c4c1f06c0069d9fa221fbf25c41c3eb`
- submission SHA: inference 未実行
- rerun check: 未実行

## push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active rule variant 数: 6
- LightGBM config 数: 0
- fold 数: 0
- total boosters: 0
- parent/control 再学習: なし
- expected train unknown rows: exp072 と同程度の約 3.78M rows

## コマンドログ

```bash
make new-steering EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
make new-exp EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
```

- result: steering / experiment scaffold 作成。

```bash
.venv/bin/ruff format experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
.venv/bin/ruff check experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
.venv/bin/python -m py_compile \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/settings.py
```

- result: PASS

## v4 test known TVT direct fit 実装

ユーザー指示: test の known TVT で `dTVT ~= a*dZ+b` を fit し、それを未知 test suffix に transform して予測する形式にする。

実装:

- `known_tvt_fit_full` variant を追加。
- `known_tvt_fit_full` は query/test well の `TVT_input` が存在する全 known rows だけで `dTVT ~= a*dZ+b` を最小二乗 fit する。
- unknown suffix の true `TVT` は fit / assignment / prediction に使わない。
- last known `TVT_input` を anchor とし、unknown suffix の各 row を `TVT_next = TVT_prev + a*dZ + b` で累積予測する。
- fit 不能な well だけ train source full-fit median `a,b` に fallback する。
- `config.yaml` の `model.params.selected_variant` と `inference.selected_variant` を `known_tvt_fit_full` に変更。

Kaggle 実行状態:

- train: 未実行
- inference: 未実行
- submission: 未提出

Kaggle v4 push 前コスト:

- active rule variants: 1 (`known_tvt_fit_full`)
- LightGBM config: 0
- folds: 0
- total boosters: 0
- GPU: なし
- parent/control retraining: なし
- purpose: test/query known `TVT_input` fit direct path の CV/LB 確認のみ。v1-v3 の比較 variants は履歴として残し、v4 Kaggle run では再実行しない。

ローカル検証:

```bash
.venv/bin/python -m py_compile experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/settings.py
.venv/bin/ruff check experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py --select F821,F401
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
.venv/bin/python scripts/validate_experiment.py --experiment exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
```

- result: PASS
- config selected variant resolve: `known_tvt_fit_full` is in `VARIANT_SPECS`.
- function smoke: known `TVT_input` から fit した係数で unknown suffix を last known anchor から累積し、expected prediction に一致した。定数 `dz` の小例では `a,b` は非一意だが、`a*dZ+b` の変換値は一致する。

## v4 Kaggle 実行結果

Kaggle package:

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-train --title 'exp206 dz dtvt bpeak cluster train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline \
  EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-inference --title 'exp206 dz dtvt bpeak cluster inference' --run-on-push --strict"
```

Metadata:

- train/inference both `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
- active rule variants: 1 (`known_tvt_fit_full`)
- LightGBM config 0、fold 0、total boosters 0

Train:

```bash
make push-kaggle-train EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp206-dz-dtvt-bpeak-cluster-train
```

- train kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v4
- selected variant: `known_tvt_fit_full`
- CV RMSE: 52.50742292458995
- MAE: 32.70950823131668
- within10: 0.34398276527759464
- bias: -4.766830325030437
- train OOF decompressed SHA: `909d533512f688f8f5bcb3848cf72c55e0493b310e335b691da3bf1540bf46d7`
- train OOF raw gzip SHA: `f724b6b773f41bc310ff119fb942c97d523926f7b1169d92e3203803ba60fd6d`
- full fit SHA: `8dc878046a32689fc5d4db485dc4b313054b186654baf2071bea300398eacf45`
- variant metrics SHA: `4883b0829be9e440e6a0719364bbd4e66da1ef30bc79228617a156664a7b08b1`

Inference:

```bash
make push-kaggle-infer EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp206-dz-dtvt-bpeak-cluster-inference
```

- inference v4 failed with `ValueError: No kernel name found in notebook and no override provided.`
- root cause: inference `.ipynb` lost kernelspec metadata after Jupytext conversion.
- fix: add Jupytext header with `kernelspec.name=python3` to `exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py`, reconvert `.ipynb`, re-prepare package.
- inference kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-inference` v5
- selected variant: `known_tvt_fit_full`
- uses test known TVT direct fit: true
- uses test tail true TVT: false
- submission rows: 14,151
- prediction range: 11576.373542877029 - 12225.250300208621
- submission SHA: `ba1903650c6da55cd64656e0eedc475701494e7250b0c62e0dd7d28a84f5e5d2`
- test assignments SHA: `a7f990fdc1a01aae6f477053e4371e1102e15af4c23684eee4b1ac1aef958d4e`

Submit:

```bash
make kaggle-output KERNEL=kentookumura/exp206-dz-dtvt-bpeak-cluster-inference \
  OUT=/tmp/kaggle-output/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/inference_v5
make submit-check EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline \
  SUBMISSION=/tmp/kaggle-output/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/inference_v5/submission.csv
make submit-code KERNEL=kentookumura/exp206-dz-dtvt-bpeak-cluster-inference \
  KERNEL_VERSION=5 OUTPUT_FILE=submission.csv MESSAGE="exp206 v4 known_tvt_fit_full"
```

- submit-check: PASS
- submission ref: `54458212`
- Public LB: 57.063
- Private LB: -

判断:

- v4 は「test known TVT で `dTVT ~= a*dZ+b` を fit して unknown suffix に transform」の指定形式そのものを実行できた。
- CV 52.5074 / Public LB 57.063 で、v3 の CV 35.3004 / Public LB 29.193 から悪化した。
- known prefix 全体の線形 fit は hidden tail の長い外挿に耐えないため、採用しない。

Kaggle train v3:

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-train --title 'exp206 dz dtvt bpeak cluster train' --run-on-push --strict"
make push-kaggle-train EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
kaggle kernels status kentookumura/exp206-dz-dtvt-bpeak-cluster-train
kaggle kernels logs kentookumura/exp206-dz-dtvt-bpeak-cluster-train
```

- kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v3
- status: COMPLETE
- runtime log time: 約 1,220 sec
- active rule variants: 15
- LightGBM config / fold / booster: 0 / 0 / 0
- best/selected variant: `discussion_fullxyz_cluster_holdout_ab_k24_h300`
- CV RMSE: 35.30041735041327
- MAE: 22.420539999938363
- within10: 0.4311791075502598
- bias: -0.3663709611314556
- train rows / wells: 3,783,989 / 773
- b peak centers: `[-0.031180282755315196, 0.030967642042632977]`
- train OOF decompressed SHA: `84bca0923a06be2689dc2b44e6a275dd00d262a0a5add55744e7a45a9630b841`
- train OOF raw gzip SHA: `ac5c913d323865cd3e70518b188751620790396f04f7b42a64fb7381ed324835`
- full fit SHA: `8dc878046a32689fc5d4db485dc4b313054b186654baf2071bea300398eacf45`
- variant metrics SHA: `3a9a3e75b767a162c579d47109f4a28256163710f2d7fa79194d0f34b3444779`
- output archive: 未取得。logs に CV / SHA が出ており、OOF gzip が 461,838,042 bytes と大きいため。

Kaggle inference v3 / submit:

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-inference --title 'exp206 dz dtvt bpeak cluster inference' --strict"
make push-kaggle-infer EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
kaggle kernels output kentookumura/exp206-dz-dtvt-bpeak-cluster-inference -p experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/kaggle/output/inference_v3
.venv/bin/python .agents/skills/kaggle-submit-check/scripts/check_submission.py experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/kaggle/output/inference_v3/submission.csv --sample data/raw/sample_submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp206-dz-dtvt-bpeak-cluster-inference -v 3 -f submission.csv -m "exp206 v3 discussion711308 full xyz last300 cluster ab baseline"
kaggle competitions submissions rogii-wellbore-geology-prediction
```

- inference kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-inference` v3
- selected variant: `discussion_fullxyz_cluster_holdout_ab_k24_h300`
- inference status: COMPLETE
- submission rows: 14,151
- prediction range: 11601.865938 - 12229.822692
- prediction mean/std: 11906.610028 / 270.414472
- submission SHA: `0f37a593ff2a3cf3ffedcf4ecfadcaaae5d6d2ac68ba86b564eb43f608afdd23`
- test assignment SHA: `7fbbccaea153eda1dbe19f5fbba7c7782429e0fb4b668ed40f5a39bfc328495f`
- submit-check: PASS
- submission ref: `54408573`
- Public LB: `29.193`

結論:

- v3 は v2 の Public LB 34.908 から 29.193 へ改善した。
- ただし要件の LB 約 12.8 にはまだ大きく届かない。
- full X/Y/Z geometry + last-300 TVT/XYZ shape cluster は、v2 の feature-nearest / fixed-a selector より hidden scoring には合ったが、standalone direct baseline としては不採用。

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --set-kernel python3 \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.ipynb
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --set-kernel python3 \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.ipynb
```

- result: PASS。notebook metadata に `kernelspec.name=python3` を設定。

```bash
.venv/bin/python scripts/validate_experiment.py --experiment exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
```

- result: PASS

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp206-discussion711308-dz-dtvt-bpeak-cluster-baseline-train --title 'exp206 discussion711308 dz dtvt bpeak cluster baseline train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp206-discussion711308-dz-dtvt-bpeak-cluster-baseline-inference --title 'exp206 discussion711308 dz dtvt bpeak cluster baseline inference' --strict"
```

- result: PASS
- train metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`, `kernel_sources=[]`
- inference metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=false`, `kernel_sources=[]`

## 実装完了時点の状態

- 状態: implemented / pending Kaggle train
- Kaggle train / inference は未 push。
- output / metrics は未生成。
- `KAGGLE_DIRECTION.md` の backlog は結果待ちとして残し、実行完了後に完了/不採用/支持を判断する。

## Kaggle train push

初回は full experiment slug を使って prepare / push した。

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp206-discussion711308-dz-dtvt-bpeak-cluster-baseline-train --title 'exp206 discussion711308 dz dtvt bpeak cluster baseline train' --run-on-push --strict"
make push-kaggle-train EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
```

- result: `SaveKernel` 400 Bad Request
- metadata 上は title slug と id slug が一致していた。
- slug length が 60 文字で既存 package より長いため、Kaggle 側の slug 長制限または SaveKernel validation の可能性が高いと判断。
- 同じ exp のまま、意味を保った短い slug に寄せて再 prepare した。

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-train --title 'exp206 dz dtvt bpeak cluster train' --run-on-push --strict"
make push-kaggle-train EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
```

- result: `Kernel version 1 successfully pushed`
- kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp206-dz-dtvt-bpeak-cluster-train`
- pulled metadata: `id_no=126138925`, CPU, internet off, competition source `rogii-wellbore-geology-prediction`

```bash
kaggle kernels logs kentookumura/exp206-dz-dtvt-bpeak-cluster-train
kaggle kernels status kentookumura/exp206-dz-dtvt-bpeak-cluster-train
```

- initial logs: empty。実行中 logs が空のことは既知なので失敗扱いにしない。
- initial status: `KernelWorkerStatus.RUNNING`

追加監視:

```bash
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp206-dz-dtvt-bpeak-cluster-train
kaggle kernels status kentookumura/exp206-dz-dtvt-bpeak-cluster-train
kaggle kernels logs kentookumura/exp206-dz-dtvt-bpeak-cluster-train
```

- `logs -f`: 数分間 stdout 空。手動停止。
- status 再確認: `KernelWorkerStatus.RUNNING`
- logs 再確認: empty
- ユーザー指示により監視停止。完了連絡後に logs / output / metrics / result を記録する。

## Kaggle train 完了確認

ユーザー完了連絡後に status / logs を再確認した。

```bash
kaggle kernels status kentookumura/exp206-dz-dtvt-bpeak-cluster-train
kaggle kernels logs kentookumura/exp206-dz-dtvt-bpeak-cluster-train
```

- status: `KernelWorkerStatus.COMPLETE`
- kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v1
- runtime: Kaggle log time 約 441 sec で train audit summary 出力
- output archive: 未取得。logs に metrics / SHA / artifact size が出ており、巨大な `exp206_train_oof_predictions.csv.gz` 209,642,039 bytes を含むため、現時点では丸ごと download しない。

主要結果:

| variant | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `nearest_xy_k8` best | 58.74433606326769 | 36.67168456216196 | 0.3185323741691638 | 1.3833495370616558 |
| `exact_typewell_peak_xy_k8` selected | 81.7364272463997 | 34.15559469912352 | 0.532409317257529 | -3.305140523213267 |

その他:

- train rows: 3,783,989
- train wells: 773
- b peak centers: `[-0.031180282755315196, 0.030967642042632977]`
- b peak count: 2
- feature / decompressed OOF SHA: `5940a36cb3682760ed6e7cebfc2545034695314ae0f0236454e659451548e5e6`
- raw gzip OOF SHA: `6b4714927fd37042dbd942cf61da09a72c4c1f06c0069d9fa221fbf25c41c3eb`
- full fit SHA: `8f0b2550ab3fba3e789f2ca76f82330075df9b86021e7cf19f76bda3f8fb97c4`
- variant metrics SHA: `030ea57eebe32e4aeed6214a0494fb0029360346bd0a34b1e85c2607ead7252c`

判断:

- selected variant は RMSE 81.736、best variant でも RMSE 58.744 で、discussion 711308 の direct baseline 再現としては成立しない。
- exact typewell / b-peak / XY nearest の target-free assignment だけでは tail drift と level offset を安定して説明できない。
- この時点では inference / submit へ進めない判断にしたが、後続でユーザー指摘により LB 約 12.8 達成要件を直接確認するため inference / submit を実行した。
- `b` peak は direct TVT candidate ではなく、使うとしても offset / confidence diagnostic の材料に限定する。

## Kaggle inference / submission

ユーザー指摘: 「要件である LB~12.8 を達成できていない」。この指摘は正しいため、完了扱いを取り消し、同じ exp206 のまま inference と code submission を実行した。

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-inference --title 'exp206 dz dtvt bpeak cluster inference' --run-on-push --strict"
make push-kaggle-infer EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
kaggle kernels output kentookumura/exp206-dz-dtvt-bpeak-cluster-inference -p experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/kaggle/output/inference_v1
.venv/bin/python .agents/skills/kaggle-submit-check/scripts/check_submission.py experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/kaggle/output/inference_v1/submission.csv --sample data/raw/sample_submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp206-dz-dtvt-bpeak-cluster-inference -v 1 -f submission.csv -m "exp206 discussion711308 dz dtvt bpeak cluster baseline"
kaggle competitions submissions rogii-wellbore-geology-prediction
```

- inference kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-inference` v1
- inference status: `KernelWorkerStatus.COMPLETE`
- submission rows: 14,151
- test wells: 3
- prediction range: 11576.993655 - 12229.053203
- submission SHA256: `46832ec24b291f0fb1d6e6ecf2f1a29879da334ee83c10382734334f248f41a3`
- test assignment SHA256: `af661f82988cb8dcf729ca173a1010ecab3470c0f490c6ac673792d7d09281db`
- submit-check: PASS
- submission ref: `54395246`
- Public LB: `41.214`

結論:

- 要件である Public LB 約 12.8 は達成できなかった。
- exp206 の実装は discussion 711308 の再現として失敗。
- 特に `dTVT/dMD = a*dZ/dMD + b` として rate fit し、source full-fit `a,b` を exact typewell / b-peak / XY nearest で割り当てる現在の実装は、hidden scoring で大きく崩れる。
- 再挑戦するなら、discussion 本文の `dTVT ≈ a*dz + b` を step/row increment として扱う variant、X-Y-Z + last-300 TVT の clustering そのもの、local wells / cluster の source selection を別設計として作り直す。

## 完了状態

- `metrics.json`、`result.md`、`README.md`、`experiment_summary.md` を submitted / Public LB 41.214 / 要件未達として更新済み。
- `SUBMISSIONS.md` に ref `54395246` を追加済み。
- `KAGGLE_DIRECTION.md` の `discussion711308_dz_dtvt_bpeak_cluster_baseline` backlog を、要件未達の失敗結果として判断メモへ移動済み。
- helper の train metrics status を `train_audit_completed` へ更新し、Kaggle package train/inference を短い slug で再 prepare 済み。再 push はしていない。
- `scripts/validate_experiment.py --experiment exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline` は PASS。

## v2 step-fit / prefix-holdout 再実装

ユーザー指摘どおり v1 は要件 LB 約 12.8 を達成していなかったため、同じ exp206 のまま再実装した。

変更内容:

- fit equation を discussion 本文に合わせて `dTVT = a*dZ + b` の row-step increment に変更。
- inference 積分も `a*dZ + b` を累積する形に変更し、v1 の `b*dMD` を廃止。
- X/Y/Z + last-300 TVT/Z slope/delta の feature-nearest source selection を追加。
- visible prefix の末尾を holdout にして source `b` / source `a,b` candidate を選ぶ selector を追加。
- active rule variants は 10、LightGBM config 0、fold 0、booster 0、GPU なし。

ローカル smoke:

```bash
.venv/bin/python -m py_compile experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/settings.py
.venv/bin/ruff check experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
.venv/bin/python scripts/validate_experiment.py --experiment exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
```

- result: PASS

Kaggle train v2:

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-train --title 'exp206 dz dtvt bpeak cluster train' --run-on-push --strict"
make push-kaggle-train EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp206-dz-dtvt-bpeak-cluster-train
```

- kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v2
- status: COMPLETE
- runtime log time: 約 700 sec
- selected at train package time: `step_xyz_tvt_tail_k8`
- best variant: `prefix_holdout_source_b_fixeda_h600`
- best CV RMSE: 35.41055512960111
- selected `step_xyz_tvt_tail_k8` CV RMSE: 35.79702583716852
- train rows / wells: 3,783,989 / 773
- b peak centers: `[-0.031180282755315196, 0.030967642042632977]`
- train OOF decompressed SHA: `51648ed1e36b6f193fa625037a005a6add6a69f0b09ff5bce308ac420cc79669`
- train OOF raw gzip SHA: `5596d475e7f4b505e506cae41e4fc66cf805f6a4ec67d402700c80068b7a67ed`
- full fit SHA: `e61b45b0dceb0f4354f9882e2cb773b9d538a54331633290e3f9c0e3f0287644`
- variant metrics SHA: `4efc9eadbb04ca4fbc5fcf118a112f685e850b4926e47ea9e0cbc15196f17799`
- output archive: 未取得。logs に CV / SHA が出ており、OOF gzip が 327,540,424 bytes と大きいため。

Kaggle inference v2 / submit:

```bash
make prepare-kaggle-notebooks EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp206-dz-dtvt-bpeak-cluster-inference --title 'exp206 dz dtvt bpeak cluster inference' --run-on-push --strict"
make push-kaggle-infer EXP=exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
kaggle kernels output kentookumura/exp206-dz-dtvt-bpeak-cluster-inference -p experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/kaggle/output/inference_v2
.venv/bin/python .agents/skills/kaggle-submit-check/scripts/check_submission.py experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/kaggle/output/inference_v2/submission.csv --sample data/raw/sample_submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp206-dz-dtvt-bpeak-cluster-inference -v 2 -f submission.csv -m "exp206 v2 discussion711308 step fit prefix holdout baseline"
kaggle competitions submissions rogii-wellbore-geology-prediction
```

- inference kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-inference` v2
- selected variant: `prefix_holdout_source_b_fixeda_h600`
- inference status: COMPLETE
- submission rows: 14,151
- prediction range: 11604.795958 - 12227.509684
- prediction mean/std: 11902.937271 / 266.428164
- submission SHA: `fcd44d9ada12214d605eaf301751b6ff932e27de9992008be459e8fd537fed4c`
- test assignment SHA: `6d410493bd4416269d9f1378af1bd82f2c377124a1756ce1c9ee730b2b2546f8`
- submit-check: PASS
- submission ref: `54396544`
- Public LB: `34.908`

結論:

- v2 は v1 の Public LB 41.214 から 34.908 へ改善したが、要件の LB 約 12.8 には大きく届かない。
- discussion 本文に寄せた step-fit、X/Y/Z + last-300 cluster、prefix-holdout selector でも standalone direct baseline としては成立しなかった。
- exp206 は「実装・Kaggle 実行・提出済み、要件未達」として閉じる。

## 再現失敗原因調査

ユーザー依頼により、LB 約 12.8 を再現できない原因をローカル train pseudo-tail で切り分けた。Kaggle 再実行ではなく、`data/raw/train` の 773 wells / 3,783,989 hidden rows を使い、oracle 係数と現行 source selector を同じ評価面で比較した。

主要結果:

| diagnostic | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| hidden suffix oracle, free `a,b` | 5.342146 | 3.730625 | 0.934881 | -0.087479 |
| hidden suffix oracle, `a=-1` fixed + best `b` | 7.587489 | 5.284173 | 0.853704 | -0.162410 |
| target full true fit `a,b` oracle | 12.379107 | 8.632166 | 0.686484 | -0.913305 |
| target full true fit, fixed `a=-1` step `b` | 20.225540 | 13.545775 | 0.533813 | -0.499125 |
| diagnostic feature16 holdout with fixed-`a` source step `b` | 34.895049 | 22.422878 | 0.416903 | -1.883085 |
| v2 selected `prefix_holdout_source_b_fixeda_h600` | 35.410555 | 22.125611 | 0.425826 | -2.173651 |
| v2 `step_xyz_tvt_tail_k8` | 35.797026 | 20.873907 | 0.447931 | -0.588541 |
| own prefix tail300 fixed `a=-1` step `b` | 40.028805 | 25.067467 | 0.418119 | -0.709543 |
| own prefix tail300 free `a,b` | 62.856780 | 32.495263 | 0.389354 | 0.833513 |

`b` 診断:

- v2 selected の assigned `b` は hidden tail の fixed-`a` oracle `b` に対して corr 0.949598、weighted mean abs error 0.008150。
- target full true free-`a,b` の `b` は hidden fixed-`a` oracle `b` に対して weighted mean abs error 0.002830 で、RMSE 12.379107 まで届く。
- prefix tail300 free-`a,b` の `b` error は weighted mean abs 0.019970、prefix tail300 fixed-`a` step `b` error は 0.009339。
- hidden rows は median 4,840、p90 6,348.6。`b` error 0.01 は median tail で約 48.4 ft、p90 tail で約 63.5 ft の累積ドリフトになる。
- prefix/full `b` peak label match は unweighted 86.93%、hidden-row weighted 86.47%。peak label は大局 signal としてはあるが、要件水準に必要な連続 `b` 精度には足りない。
- prefix holdout RMSE は mean 1.515657 / p90 3.479729 と良く見えるが、hidden suffix の RMSE 35.410555 と乖離しており、visible prefix 後半の最良 `b` が long hidden tail に外挿できていない。

判断:

- `dTVT = a*dZ+b` の式自体は失敗原因ではない。hidden oracle では 5.34-7.59、target full true fit でも 12.38 で、discussion 711308 の LB 約 12.8 に近い上限は存在する。
- Kaggle v2 の train CV 35.410555 と Public LB 34.908 が揃っているため、submission CSV、kernel packaging、sample row alignment の問題ではない。
- 主原因は target-free な source / `b` selector の精度不足。現行の X/Y/Z + last-300 feature-nearest、prefix holdout、b-peak coarse cluster では、long tail で必要な `b` error 0.002-0.003 台に届かない。
- source `b` を `a=-1` 固定 fit に揃えた診断 variant は RMSE 34.895049 で、現行 selected 35.410555 から小改善に留まった。自由 `a,b` fit の `b` を fixed-`a` prediction に流用している不整合は副次要因で、主原因ではない。
- 再挑戦は exp206 の小修正ではなく、discussion 699853 の discrete offset selector / classifier、または PF/Beam / ML selector へ `b` oracle gap を特徴量として渡す別設計が必要。

## v3 discussion literal cluster 修正

ユーザー指摘: discussion 711308 の本文「I tried clustering test wells by X-Y-Z as well as last-300 TVT, etc., and fitting dz to dTVT using a and b parameters from local wells / cluster. Baseline result was LB ~ 12.8.」に対して、v2 selected は `feature_nearest` と `a=-1` 固定 + source `b` 選択であり、同じ内容とは言い切れなかった。

同じ exp206 のまま、ディスカッション文に近づける修正を入れた。

- last-300 TVT/XYZ を slope/delta だけでなく 6 点の relative shape samples として追加。
- test/train とも tail true TVT を使わず、全 well で利用可能な full X/Y/Z geometry shape samples を追加。
- source wells 上で deterministic k-means 風 cluster を作り、query/test well を最寄り cluster center に割り当てる `feature_cluster` mode を追加。
- cluster/local source の `a,b` を両方使う `discussion_cluster_ab_*` / `discussion_fullxyz_cluster_ab_*` variants を追加。
- cluster 内 source `a,b` を visible prefix holdout で選ぶ `discussion_fullxyz_cluster_holdout_ab_k24_h300` を selected にした。

ローカル pseudo-tail 比較:

| variant | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `discussion_fullxyz_cluster_holdout_ab_k24_h300` | 35.300417 | 22.420540 | 0.431179 | -0.366371 |
| `prefix_holdout_source_b_fixeda_h600` v2 selected | 35.410555 | 22.125611 | 0.425826 | -2.173651 |
| `discussion_fullxyz_cluster_ab_k24` | 35.464889 | 21.395210 | 0.440976 | -2.147272 |
| `discussion_cluster_holdout_ab_k24_h300` | 37.719929 | 23.465085 | 0.426230 | -1.194100 |
| `discussion_cluster_ab_k24` | 37.738513 | 22.565128 | 0.425689 | -0.741718 |
| `discussion_cluster_ab_k12` | 44.427756 | 24.509940 | 0.430302 | -0.801845 |

判断:

- v3 は v2 よりディスカッション本文に近い明示的 cluster 実装になった。
- full X/Y/Z geometry を入れない cluster は v2 より悪化したため、ディスカッションの X-Y-Z は last-known 周辺ではなく full well geometry と解釈する方が妥当。
- full X/Y/Z geometry + last-300 TVT/XYZ shape の cluster は v2 よりわずかに改善したが、RMSE 35.300417 で、LB 約 12.8 の再現にはまだ遠い。
- Kaggle push 前コスト: active rule variants 15、LightGBM config 0、fold 0、total boosters 0、GPU なし、parent/control 再学習なし。

ローカル検証:

```bash
.venv/bin/python -m py_compile experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/settings.py
.venv/bin/ruff check experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/dz_dtvt_bpeak_cluster_baseline.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py --select F821,F401
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline/exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.py
.venv/bin/python scripts/validate_experiment.py --experiment exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline
```

- result: PASS
