# exp221_lgb_oof_gaussian_emission_hmm_on_exp148

## 状態

Kaggle train v3 完了。`sigma=20/lambda=0.50` の HMM+LGB single variant は train-side OOF RMSE 8.327736951 で、exp148 `lgb_mean` 8.501290984 と exp193 `lgb_mean` 8.456676053 を上回った。

Kaggle inference v1 も完了。`kentookumura/exp221-lgb-hmm-exp148-infer` v1 で exp148 saved `lgb_mean` current-test prediction を notebook 内生成し、同じ HMM candidate で `submission.csv` を作成。14,151 rows、fallback 0、submission validation PASS。

Code submission v1 は ref `54490473` で Public LB 7.953。exp148 GPU 7.960 は小さく上回ったが、exp193 7.946、exp148 CPU runtime 7.921、exp218 ML anchor 7.843、exp082 ensemble anchor 7.601 には届かないため採用しない。

## 仮説

exp148 / exp193 の LightGBM OOF 点予測を、exp209 exact HMM の state TVT に対する Gaussian emission として追加すれば、ML の局所的な強さと HMM の系列平滑化を同じ posterior 内で統合できる可能性がある。

## 検証方針

初回は exp148 `lgb_mean` OOF を Gaussian center とし、Kaggle v2 の 3 variant run が timeout したため、v3 では partial logs で最良だった `sigma=20/lambda=0.50` のみを実行した。新規 LightGBM 学習、exp072 再生成、raw-test inference、submit は行っていない。

比較対象は exp148 `lgb_mean`、exp193 `lgb_mean`、exp072 `likpf_mean`。overall RMSE、distance bucket、worst-well、exp115 hidden-like split、HMM std calibration、step-delta smoothness を確認した。

## 所見

Train-side では採用候補。overall、全 distance bucket、hidden-like subgroup、step-delta で exp148 / exp193 を改善した。一方で exp148 比 264 wells、exp193 比 278 wells は悪化しており、最大悪化 well は `2e63d9de`。

Inference v1 は形式面では通っている。見えている test は sample なので score 判断には使わず、確認点は current-test exp148 予測の notebook 内生成、strict sample id coverage、fallback 0、submission validator PASS。

LB は train-side の大きな改善ほど伸びなかった。fixed sigma の HMM smoothing は hidden 側では exp148 GPU に対する小幅改善に留まり、現時点の提出 anchor にはしない。

## 参照ファイル

- `config.yaml`
- `metrics.json`
- `result.md`
- `exp221_lgb_oof_gaussian_emission_hmm_on_exp148_inference.py`
- `exp221_lgb_oof_gaussian_emission_hmm_on_exp148_train.py`
- `exact_hmm_smoother.py`
- `direct_hmm_comparison.py`
