# exp288_known_tvt_typewell_horizontal_gr_visualization セッションノート

## 目的

全train wellでknown区間は`TVT_input`、prediction-target区間はtrain true `TVT`を使って
Type Well `TVT -> GR`を線形補間し、full-well参照GRとhorizontal GRを上下2段で比較するPNGを保存する。

## 現在の状態

- Route: pf_beam
- 状態: Kaggle train v1 complete / 全773 PNG取得・全件SHA検証完了
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
make new-steering EXP=exp288_known_tvt_typewell_horizontal_gr_visualization
make new-exp EXP=exp288_known_tvt_typewell_horizontal_gr_visualization
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/exp288_known_tvt_typewell_horizontal_gr_visualization_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/exp288_known_tvt_typewell_horizontal_gr_visualization_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/exp288_known_tvt_typewell_horizontal_gr_visualization_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/exp288_known_tvt_typewell_horizontal_gr_visualization_inference.py
.venv/bin/python -m py_compile experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/*.py
.venv/bin/ruff check experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/*.py --select F821,F401,E9
.venv/bin/python scripts/validate_experiment.py --experiment exp288_known_tvt_typewell_horizontal_gr_visualization
make prepare-kaggle-notebooks EXP=exp288_known_tvt_typewell_horizontal_gr_visualization EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train --title 'exp288 known tvt typewell horizontal gr viz train' --strict --no-src"
make prepare-kaggle-notebooks EXP=exp288_known_tvt_typewell_horizontal_gr_visualization EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train --title 'exp288 known tvt typewell horizontal gr viz train' --run-on-push --strict --no-src"
make push-kaggle-train EXP=exp288_known_tvt_typewell_horizontal_gr_visualization
kaggle kernels pull kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train -p /tmp/kaggle-pull-exp288-v1 -m
kaggle kernels status kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train
kaggle kernels logs kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train
kaggle kernels output kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train/1 -p experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/kaggle/output/train_v1 --quiet
```

### 検証結果

- 最初のJupytext commandはexperiment directoryをcwdにしたため相対`.venv/bin/jupytext`が見つからず失敗。
  repo rootから同じ変換を再実行して成功した。Notebookコードやデータ処理には到達していない。
- train / inference Jupytext変換・`--test`: PASS。kernelspecは`python3`。
- `py_compile`: PASS。
- Ruff `F821,F401,E9`: PASS。
- strict experiment validation: PASS。
- train notebook: 15 cells / 645行 / 7章。親exp168 trainは1,716行 / 8章だが、exp288は
  shift-scan、OOF join、複数plot familyを持たない単純なall-well interpolation diagnosticであり、
  runtime/config、input、interpolation、plot/index、setup、execution、summaryを全てセル上に展開した。
- inference notebook: 5 cells。診断専用のためprediction/submissionを作らないno-opを明記。
- raw train preflight: horizontal 773 files / typewell 773 files / missing pair 0。
- Kaggle package: `experiments/exp288_known_tvt_typewell_horizontal_gr_visualization/kaggle/train`。
- kernel metadata: private / CPU / GPU off / TPU off / internet off / `run_on_push=false` / competition source 1件 / kernel source 0件。
- canonical id/title: `kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train` /
  `exp288 known tvt typewell horizontal gr viz train`。slug一致。
- loose/package config SHA: `a9fa565d...b93c`で一致。
- loose/package train source SHA: `157ff807...8a00`で一致。
- bootstrap manifest: config、train/inference source、settings、projectの5 files。埋め込みconfig/source SHAも一致。
- この初期検証時点ではKaggle push/run、local notebook run、PNG生成は実行していない。

### Prediction-target EDA拡張の検証

- known rowは`TVT_input`、prediction-target rowはtrain true `TVT`を補間座標に使うfull-well表示へ更新した。
- `TVT_input.isna()`の連続MD区間を両パネルで黄色着色し、図、HTML index、manifest、summaryに
  train-only true TVT EDAであることを明記した。
- train Jupytext変換・`--test`: PASS。15 cells、kernelspec `python3`。
- `py_compile`: PASS。Ruff `F821,F401,E9`: PASS。strict experiment validation: PASS。
- 既知2 row + target 2 rowのsynthetic testで、補間参照GRとtarget MD spanを検証: PASS。
- synthetic testの初回試行はlocal venvに`matplotlib`がないためimport時に停止した。Notebookや実データは実行していない。
- Kaggle packageを同じcanonical id/title、private CPU、`run_on_push=false`で再prepareした。
- 更新後のloose/package config SHA:
  `fa4b379eab790c4eba868f5b5e603d6d1c603d7800590f55c18c371a55911f4d`で一致。
- 更新後のloose/package train source SHA:
  `8031c9309bcc906a3578eded74c0351db52d27816c3a86c4c530872fb5d19a72`で一致。
- この拡張検証時点ではKaggle push/run、local notebook run、PNG生成は実行していない。

## 2026-07-19 Kaggle train v1 完了記録

- 実行契約: private CPU、GPU/TPU/internet off、active variant / model config / fold / booster =
  0 / 0 / 0 / 0。control / parent retraining、PF/Beam生成、inference、submissionなし。
- canonical kernel: `kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train`。
- Kaggle kernel version: 1。id_no: `127877148`。status: `COMPLETE`。
- Kaggle URL: `https://www.kaggle.com/code/kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train`。
- 入力well: 773。保存PNG: 773。skip: 0。skip reason: なし。
- Notebook実処理時間: `388.99937748908997`秒。
- manifest CSV SHA256:
  `8cd742699beb4a649247d68b533fb58a2fd1c492e51bdadb9c2ca5cdd0b68eb1`。
- lazy-load HTML index SHA256:
  `1ebdddeb7fcc1aacd51de21d5797cf9c0aab0cc0d5ea48c507ee97f532bba20c`。
- 全outputを`kaggle/output/train_v1`へ取得した。PNG 773枚の合計は157,474,395 bytes。
- manifest 773行に対して、saved status、PNG存在、byte size、SHA256を全件照合した。
  missing / size mismatch / SHA256 mismatch = 0 / 0 / 0。
- 代表PNG `000d7d20.png`を目視し、上段Type Well参照GR、下段horizontal GR、full-well MD軸、
  prediction-target区間の黄色着色を確認した。
- stderrはPython debuggerとnbconvert dependencyのwarningのみで、Notebook errorはない。
- quality metricsは空配列のまま。residual scale、NCC、affine、自己相関、entropyは計算していない。
- local notebook runは実行していない。

## 変更点

- `exp168`の16-well shift-scan visualizationを上書きせず、新しい全well known-TVT評価面を作る。
- Type Wellのfinite `TVT,GR`をTVT順に並べ、重複TVTはGR medianへ集約する。
- Type Well TVT範囲外を外挿せず、horizontal GR欠損も補完しない。
- 1 well 1 PNG、manifest CSV、lazy-load HTML index、summary JSONを保存する。
- 初回はresidual scale、NCC、affine、自己相関、entropyを推定しない。
- ユーザー追加指定により、prediction-target rowでもtrain true `TVT`から参照GRを生成し、full-well表示へ拡張する。
- `TVT_input.isna()`区間は両パネルを着色し、図/HTML/summaryにtrue TVT使用を明記する。
- target true TVTはtrain-only EDA図示専用で、特徴量・学習・selector・inference・submissionへ使わない。
- active variant / LightGBM config / fold / booster: 0 / 0 / 0 / 0。
- control / parent retraining、PF/Beam生成、GPU、inference、submission: すべてなし。

## 再現性メモ

- seed policy: RNGなし、well名辞書順、single process。
- stochastic components: なし。
- CPU/GPU runtime: CPU only、GPU/internet off。
- Kaggle kernel id / version: `kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train` / 1。
- input / feature schema SHA: manifestに入力path・bytesを保存した。feature schema対象外。
- feature content SHA: manifest CSV / HTML indexのSHA256を記録し、全PNGをmanifest SHAへ照合した。
  PNGは描画library差があり得るためdeterministic anchorにはしない。
- model manifest / model SHA: 対象外。
- prediction SHA: 対象外。
- submission SHA: 対象外。
- rerun check: 未実行。

## 次のアクション

1. lazy-load HTML indexで全wellの形状を目視する。
2. 定量化が必要なら、本EDAの観察を根拠にresidual/NCC/affine等を別実験として設計する。
3. prediction-target true TVTはtrain-only EDA専用のままとし、推論入力へ流用しない。
