# 要件

## 依頼

`prefix_backtest_tvt_confidence` を実装する。

## 制約

- Route: `pf_beam`
- 予測置換、hard router、submission は行わない。
- exp072 の deterministic full replay train feature cache を入力にし、true TVT は診断スコア計算にだけ使う。
- 各 well の pseudo-hidden tail を近距離 calibration phase と遠距離 holdout phase に分ける。
- expected TVT error は well-hash fold 外で推定する。holdout well の true TVT を fitting に使わない。
- GR 波形そのものは再投入せず、既存 cache の target-free confidence / disagreement scalar だけを使う。
- 再現性: `docs/06_reproducibility.md` に従い、gzip 入力は decompressed content SHA を主証拠として記録する。

## 受け入れ基準

- `experiments/exp087_prefix_backtest_tvt_confidence/` に config、train/inference notebook、補助スクリプト、記録ファイルが揃う。
- `config.yaml` に route、lineage、leakage policy、source cache、confidence feature、fold-safe audit 設定がある。
- train notebook は source schema check、audit 実行、生成物確認をセル単位で追える。
- 補助スクリプトは candidate RMSE、confidence bin、distance bucket、phase、fold、signal correlation、row-level expected error を保存する。
- `validate_experiment.py`、`ruff check`、`py_compile` が通る。
- deterministic anchor として扱わず、submission SHA / model SHA が不要な診断実験として記録されている。
