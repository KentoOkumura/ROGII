# exp132_multi_scale_gr_observation_likelihood セッションノート

## 目的

`multi_scale_gr_observation_likelihood` バックログの実装。exp099 の raw multi-observation likelihood は oracle headroom を増やしたが direct top1 / softmax blend は崩壊したため、候補を直接置換せず、PF/Beam 候補に対する target-free multi-scale GR observation likelihood と confidence / verifier feature を作る。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_rejected
- CV: train-side pseudo-tail audit best `likpf_mean` RMSE 11.594897
- LB: なし
- 提出: なし

## コマンドログ

### 2026-06-26 JST 実装

```bash
make new-steering EXP=exp132_multi_scale_gr_observation_likelihood
make new-exp EXP=exp132_multi_scale_gr_observation_likelihood SOURCE=experiments/exp099_pf_multi_observation_likelihood_probe
```

初回 `new-exp` は steering 作成との並列実行タイミングで失敗したため、steering 作成完了後に再実行して成功。

実装内容:

- `.steering/20260626-exp132-multi-scale-gr-observation-likelihood/` を作成し、requirements / design / tasklist を更新。
- `config.yaml` を train-side multi-scale GR observation likelihood audit 用に更新。
- `multi_scale_gr_observation_likelihood.py` を追加。
- train notebook を exp132 用に更新し、設定確認、入力前提、監査実行、出力 preview、metrics 保存のセル構成にした。
- inference notebook は診断専用 no-op として明記。

helper の主な処理:

- exp072 deterministic full replay train cache を固定入力として読む。
- 既存 `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` 候補を TVT absolute 値へ materialize。
- 各候補 TVT を finite prefix `TVT_input` 上の最近傍位置へ写す。
- horizontal GR から smoothed GR、local z-score、derivative、energy を window `[5, 11, 21]` で作る。
- offset `[-48, -24, -12, 0, 12, 24, 48]` の観測を比較し、MAE / NCC / z-score MAE / derivative MAE / energy MAE から score を作る。
- decoy shift `[-24, -18, 18, 24]` を比較し、decoy gap / ambiguity proxy を保存。
- `msgr_top1`、`msgr_top2`、softmax、`likpf_msgr_blend`、low-switch `msgr_gate_*` を診断候補として作る。
- candidate metrics、rank metrics、bucket metrics、by-well metrics、candidate-long、row context、wide feature cache、summary JSON を保存する。

## 予定

```bash
.venv/bin/python -m py_compile experiments/exp132_multi_scale_gr_observation_likelihood/multi_scale_gr_observation_likelihood.py experiments/exp132_multi_scale_gr_observation_likelihood/settings.py
.venv/bin/ruff check experiments/exp132_multi_scale_gr_observation_likelihood/multi_scale_gr_observation_likelihood.py experiments/exp132_multi_scale_gr_observation_likelihood/settings.py
make validate-exp EXP=exp132_multi_scale_gr_observation_likelihood
make prepare-kaggle-notebooks EXP=exp132_multi_scale_gr_observation_likelihood EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp132-msgr-likelihood-train --title 'exp132 msgr likelihood train' --run-on-push --strict"
```

### 2026-06-26 JST validation / package

```bash
.venv/bin/python -m py_compile experiments/exp132_multi_scale_gr_observation_likelihood/multi_scale_gr_observation_likelihood.py experiments/exp132_multi_scale_gr_observation_likelihood/settings.py
python3 -m json.tool experiments/exp132_multi_scale_gr_observation_likelihood/exp132_multi_scale_gr_observation_likelihood_train.ipynb
python3 -m json.tool experiments/exp132_multi_scale_gr_observation_likelihood/exp132_multi_scale_gr_observation_likelihood_inference.ipynb
.venv/bin/ruff check experiments/exp132_multi_scale_gr_observation_likelihood/multi_scale_gr_observation_likelihood.py experiments/exp132_multi_scale_gr_observation_likelihood/settings.py
make validate-exp EXP=exp132_multi_scale_gr_observation_likelihood
make prepare-kaggle-notebooks EXP=exp132_multi_scale_gr_observation_likelihood EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp132-msgr-likelihood-train --title 'exp132 msgr likelihood train' --run-on-push --strict"
.venv/bin/python -m py_compile experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/train/multi_scale_gr_observation_likelihood.py experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/train/settings.py
.venv/bin/python - <<'PY'
# /tmp に synthetic cache と raw train を作り、helper の主要 path を smoke 実行。
PY
```

結果:

- py_compile: PASS
- notebook JSON: PASS
- ruff: PASS
- validate-exp: PASS
- synthetic helper smoke: PASS。15 rows / 1 well / 75 features の feature cache と metrics 出力を確認。
- Kaggle train package: `experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/train`
- kernel id: `kentookumura/exp132-msgr-likelihood-train`
- title: `exp132 msgr likelihood train`
- metadata: GPU false / internet false / run_on_push true / source `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## 再現性メモ

- seed policy: exp132 内では新規乱数なし。
- stochastic components: upstream exp072 PF/Beam cache のみ。exp132 では再生成しない。
- GPU: 不使用。
- deterministic anchor: false。提出物や model は作らない。
- gzip 生成物は decompressed content SHA を summary JSON に記録する。

### 2026-06-26 JST Kaggle train v1 / result

```bash
make push-kaggle-train EXP=exp132_multi_scale_gr_observation_likelihood
kaggle kernels logs kentookumura/exp132-msgr-likelihood-train
kaggle kernels output kentookumura/exp132-msgr-likelihood-train -p experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/output/train_v1
cp experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/output/train_v1/metrics.json experiments/exp132_multi_scale_gr_observation_likelihood/metrics.json
```

結果:

- Kernel: `kentookumura/exp132-msgr-likelihood-train` v1
- Output: `experiments/exp132_multi_scale_gr_observation_likelihood/kaggle/output/train_v1`
- Status: completed train-side audit, rejected for follow-up
- Runtime: 2328.92 sec
- Rows / wells: 3,783,989 / 773
- Best candidate: `likpf_mean` RMSE 11.594897 / MAE 7.067633 / within10 0.772807
- Best low-switch gate: `msgr_gate_m0p08_s0p45_d40` RMSE 11.632677 / within10 0.771381
- Gate delta vs `likpf_mean`: +0.037780 RMSE
- Direct `msgr_top1`: RMSE 86.806694
- `likpf_msgr_blend_w0p1`: RMSE 14.404808
- baseline primary oracle: RMSE 7.434030 / within10 0.906525
- baseline+msgr oracle: RMSE 6.949725 / within10 0.921029 / selected_msgr_rate 0.183110
- candidate rank score top1: `beam_mean` selected, RMSE 86.806694
- Best gate by-well: 226 improved / 528 worsened / 19 same
- Feature cache rows / wells / feature_count: 3,783,989 / 773 / 75
- Feature gzip SHA256: `3ced89e1837321ea15fd22848aacde7ec8729aa7d97aae142c32fe5ff21124eb`
- Feature decompressed SHA256: `76f41392f0148d14568b87bd973d74da7c48a879ff20b1b2273a13db96606756`
- Schema SHA256: `e54c55717916aee9c71f63917f891042cd4a7b6c11df1d6cd887c3602753686f`

判定:

multi-scale GR likelihood は direct scorer、softmax / blend、low-switch gate のいずれも `likpf_mean` を上回れなかった。oracle top10 headroom はあるが、非 oracle verifier が選べず、best gate も全 distance bucket と多数 wells で悪化したため、inference port / submit / immediate exp092 add-only feature 化は行わない。
