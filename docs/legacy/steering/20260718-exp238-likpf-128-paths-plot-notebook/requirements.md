# 要件

## 依頼

Kaggle notebook `kentookumura/exp238-oof-selector-confidence-probe`
（`scriptVersionId=335655690`）の別バージョンとして、exp072 互換 likelihood-PF の
128 seed trajectory を全て半透明で重ね、true TVT と exp238 `lgb_mean` OOF を同じ
well plot 上で比較できる診断 notebook を exp238 内に追加する。

## 制約

- Route: `ml_model`。PF は exp238 ML OOF のばらつきを観察する補助診断であり、prediction blend や direct replacement は行わない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp072 v2 と同じ likelihood-PF dynamics、500 particles、128 seeds、`stable_seed("likpf", "train", well_id)` を使う。
- PF seed trajectory は全128本を描画し、線ごとに透明度を付ける。
- true TVT と exp238 final `lgb_mean_pred_tvt` OOF は PF 群より太く、不透明に描画する。
- selector、LightGBM、PF以外のcandidateの学習は行わない。submission生成・competition submitも行わない。
- 既存の selector-confidence notebook と canonical Kaggle kernel は上書きしない。
- 初回実行先は Kaggle とする。今回の依頼では package 作成までとし、push / run は行わない。

## 受け入れ基準

- Jupytext percent source、別名 `.ipynb`、CPU/internet-off の別 Kaggle package が揃う。
- 各 well plot に likelihood-PF 128本、true TVT、exp238 LightGBM OOF がある。
- 128 PF線の本数、alpha、linewidth、particles、stable seed contractをsummaryへ保存する。
- raw evaluation row IDとexp072 cache / exp238 OOFのID順をwellごとにfail-fast検証する。
- regenerated 128-seed meanと保存済みexp072 `likpf_mean`の差をwellごと・globalで監査する。
- 全128 trajectoryを大規模生成物として保存せず、well単位で解放する。PNG、manifest、plots zip、summaryだけを保存する。
- `py_compile`、`ruff --select F821`、Jupytext `--test`、notebook JSON parse、strict experiment validationが通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel versionが記録されている。
- gzip生成物を比較する場合は、raw `.csv.gz` SHAではなくdecompressed content SHAを主証拠として記録している。
