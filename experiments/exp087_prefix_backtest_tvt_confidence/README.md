# exp087_prefix_backtest_tvt_confidence

## 状態

- ルート: `pf_beam`
- 状態: `diagnostic_completed_supported`
- CV: diagnostic。primary PF RMSE 14.493051、expected-error Pearson 0.519681
- Public LB: -
- Submit: なし
- 作成日: 2026-06-20
- 親実験: `exp083_pf_beam_true_tvt_2d_well_eda`
- 入力 cache: `exp072_exp063_full_replay_feature_cache`
- Anchor: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`

## 仮説

PF/Beam 予測値は全体では ML anchor より弱いが、PF が勝つ well も多い。直接置換ではなく、PF/Beam/likelihood-PF の confidence / disagreement signal から TVT error が大きい row や bucket を fold-safe に識別できれば、後続の sample weight、residual clip、candidate ranker の材料にできる。

## 実装

- exp072 full replay train feature cache を読む。
- `true_tvt = last_known_tvt + target` に戻す。
- `beam_mean_d`、`likpf_mean_d`、`sc_ens_d` などを TVT 空間へ戻す。
- 各 well の pseudo-hidden tail を near-prefix calibration phase と holdout phase に分ける。
- calibration phase のみから `abs(pf_ancc - true_tvt)` を説明する ridge confidence model を fit し、well-hash fold 外で `expected_tvt_error` を出す。
- confidence bin、distance bucket、phase、fold、signal correlation、candidate metrics を artifact に保存する。

## 検証方針

- `validate_experiment.py` で構成と TODO 不在を確認する。
- `ruff check` と `py_compile` を通す。
- 合成 CSV で source materialize と fold-safe confidence 出力を smoke する。
- 本番 cache 読み込みと全件 metrics は Kaggle train notebook で確認する。

## 所見

Kaggle train v2 で 3,783,989 rows / 773 wells を処理した。`expected_tvt_error` は absolute error と Pearson 0.519681 で相関し、confidence bin の observed MAE は low 2.460268 から high 19.057079 まで分離した。unstable flag は全体の 20.000005% に立ち、high-error rate は stable 11.541393% に対して unstable 53.837748% だった。

上位 signal は `pf_likpf_abs`、`md_since`、`pf_beam_abs`、`beam_likpf_abs`、`likpf_delta_abs`。PF/Beam 直接置換ではなく、`pf_beam_disagreement_sample_weight` の confidence feature / sample weight 候補として使う。

## 実行入口

- Train notebook: `exp087_prefix_backtest_tvt_confidence_train.ipynb`
- Inference notebook: no-op。提出ファイルは作らない。
- Kaggle 準備:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp087_prefix_backtest_tvt_confidence --notebook train --kernel-id kentookumura/exp087-prefix-backtest-tvt-confidence-train --title "exp087 prefix backtest tvt confidence train" --run-on-push --strict
```

## 注意

この実験は confidence diagnostic であり、PF/Beam 予測値の置換や hard router の採用根拠にはしない。短い synthetic cutoff で PF/Beam を再実行する完全版ではなく、既存 exp072 cache による第一段の fold-safe calibration として扱う。
