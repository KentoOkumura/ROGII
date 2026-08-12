# exp107_selector_topn_candidate_only_features セッションノート

## 2026-06-22 実装

### コマンド

- `make new-steering EXP=exp107_selector_topn_candidate_only_features`
- `make new-exp EXP=exp107_selector_topn_candidate_only_features SOURCE=experiments/exp098_selector_rank_slot_features_on_exp073`
- `.venv/bin/python -m py_compile experiments/exp107_selector_topn_candidate_only_features/selector_topn_candidate_only_features.py experiments/exp107_selector_topn_candidate_only_features/settings.py`
- `.venv/bin/python -m json.tool experiments/exp107_selector_topn_candidate_only_features/exp107_selector_topn_candidate_only_features_train.ipynb`
- `.venv/bin/python -m json.tool experiments/exp107_selector_topn_candidate_only_features/exp107_selector_topn_candidate_only_features_inference.ipynb`
- `make validate-exp EXP=exp107_selector_topn_candidate_only_features`
- `.venv/bin/ruff check experiments/exp107_selector_topn_candidate_only_features/selector_topn_candidate_only_features.py experiments/exp107_selector_topn_candidate_only_features/settings.py`
- `make prepare-kaggle-notebooks EXP=exp107_selector_topn_candidate_only_features EXTRA_ARGS="--notebook train --run-on-push --strict"`

### 実装メモ

- `docs/legacy/steering/20260622-exp107-selector-topn-candidate-only-features/` を作成。
- exp098 を親として実験ディレクトリを作成。
- 実装ファイルを `selector_topn_candidate_only_features.py` にリネームし、出力 prefix を exp107 に変更。
- `build_selector_topn_candidate_only_features()` で、top1/top2/top3 に入った候補だけから特徴量を生成するように変更。
- `rank*_is_*`、candidate-set-wide entropy/range、全 pairwise delta、`u_corr` / `u_resid` / `u_abs_resid` / `u_fit_degree` は特徴量 group から除外。
- `config.yaml` は `top1_candidate_only`、`top2_candidate_only`、`top3_candidate_only` の 3 variant ablation に更新。
- inference notebook は train-side audit only の guard に変更。
- top-n feature group smoke で列数を確認: top1=6、top2=15、top3=21。
- `py_compile`、notebook JSON validation、`make validate-exp`、ruff は pass。
- Kaggle train package は `experiments/exp107_selector_topn_candidate_only_features/kaggle/train` に生成済み。
- train kernel id は `kentookumura/exp107-selector-topn-candidate-only-features-train`。

### 次アクション

1. Kaggle train を push / 実行する。
2. Kaggle train 実行後、OOF / worst-well / bucket / path continuity を exp098、exp105、exp092 と比較する。

## 2026-06-22 Kaggle train v1 完了

### コマンド

- `make push-kaggle-train EXP=exp107_selector_topn_candidate_only_features`
- `make prepare-kaggle-notebooks EXP=exp107_selector_topn_candidate_only_features EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp107-selector-topn-candidate-only-features-train --title 'exp107 selector topn candidate only features train' --run-on-push --strict"`
- `make push-kaggle-train EXP=exp107_selector_topn_candidate_only_features`
- `kaggle kernels logs kentookumura/exp107-selector-topn-candidate-only-features-train`
- `kaggle kernels output kentookumura/exp107-selector-topn-candidate-only-features-train -p experiments/exp107_selector_topn_candidate_only_features/kaggle/output/train_v1`

### 実行メモ

- 初回 push は Kaggle SaveKernel 400 `Your kernel title does not resolve to the specified id` で失敗した。
- kernel id は維持し、title を `exp107 selector topn candidate only features train` に短縮して package を再生成した。
- 再 push は成功。Kernel version 1。
- URL: `https://www.kaggle.com/code/kentookumura/exp107-selector-topn-candidate-only-features-train`
- output は `experiments/exp107_selector_topn_candidate_only_features/kaggle/output/train_v1` に取得済み。
- `kaggle kernels output` は予測 gzip と model files が大きく長時間無出力だったが、最終的に全出力と kernel log を取得できた。

### 結果

- rows: 3,783,989
- wells: 773
- model count: 45
- runtime: 28,860.946 sec
- best: `top2_candidate_only` / `lgb2` RMSE 9.437602823
- `top2_candidate_only` / `lgb1`: 9.437894828
- `top2_candidate_only` / `lgb_mean`: 9.479092683
- `top1_candidate_only` / `lgb_mean`: 9.577677177
- `top3_candidate_only` / `lgb_mean`: 9.527935589

### 比較

- vs exp073 raw anchor 9.526374749: -0.088771927
- vs exp077 policy 9.470514801: -0.032911978
- vs exp092 best lgb1 9.322479896: +0.115122927
- vs exp098 lgb1 9.358151052: +0.079451770
- vs exp098 lgb_mean 9.427447987: +0.010154835
- vs exp105 best 9.441103161: -0.003500339

### 監査

- best `top2/lgb2` の worst well は `86454a6f` RMSE 55.029583。
- distance bucket は 1000+ RMSE 10.347234、0-50 RMSE 1.226686。
- path continuity は step >=10 が 1、step >=25 が 0。全体崩壊はない。
- rank1 distribution は `pf_ancc` 33.65%、`beam_mean` 24.55%、`likpf_mean` 41.80%。`sc_ens` / `hyb` は rank1/rank2 に入らず、rank3 でも計 37 rows のみ。

### SHA

- summary SHA: `859fa539a92caa0b8199809ea19a99977ea1862b968da885cdfc56e4d28b664a`
- metrics SHA: `8e75e8b00fdae7d57e5874428fb249c5a2592eb311a5d07e38b58371d55dbf1f`
- feature schema SHA: `8cf8be55818ec3ea00cf19a82010c6ef185d49320f4d2402ca42ec1b59b88721`
- model manifest SHA: `26d2e5a0c3b5029ec38a1f91ba966849b9a02f96513a78ec6ba8393dabf4b412`
- predictions decompressed SHA: `8f20066736360ab8457dfdfec5cd0a42f860a9ce069bf3236f0698952757f191`

### 判断

exp107 は exp105 compact をわずかに上回ったが、exp098 full rank-slot と exp092 には届かない。追加 rank-slot 列だけを top-n candidate-only に絞る方向は rejected とし、提出しない。
