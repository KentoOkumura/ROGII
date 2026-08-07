# exp245 selector context parity on exp238

exp238 の nested candidate-error selector を、学習時と current test 推論時で同じ
context schema に直す実験です。候補集合、outer 5 / inner 4 GroupKFold、selector の
LightGBM設定は固定し、NaN fallback の原因だった特徴だけを修正します。

## 状態

Kaggle CPU train v1完了。train context parityは合格したが、worst-well +38.016697 ftで
selector safety guardは不合格。inferenceとdirect採用は停止しています。

なお、41個の`copcf_*`をtrainから除外したため、本実験は「同じ184特徴をtestでも
再生成するparity修正」ではなくfeature-removal ablationです。正しいraw-test generatorは
exp238内の`*_rawtest_copcf_parity.ipynb`で別途実装します。

## 仮説

train-only `copcf_*` を除外し、exp226診断4列をcurrent testで再生成すれば、hidden testの
well数によらずselector contextのmissing/nonfiniteを0にできる。

## 変更点

- hidden test で再生成できない train OOF 専用 `copcf_*` 41特徴を selector 学習から除外する。
- exp226 の候補値と同じ `PredictionResult` から、`geop`、`gr_delta`、
  `geop - prediction`、その絶対値を current test 上で再生成する。
- selector context を184特徴から143特徴へ変更する。
- train/inference とも context の欠損列または非有限値が1件でもあれば停止する。
- inference は保存済み20 selectorだけを読み、selectorを再学習しない。

## 実験範囲

- Route: `ensemble`
- 学習: CPU、1 config × outer 5 × inner 4 = 20 selector boosters
- 親/control/final LightGBM: 再学習なし
- direct candidate adoption / submission: この実験では行わない

exp245 の safety guard と current-test parity が通過した後、候補パスの直接採用は
別実験で評価します。

## 検証方針

outer 5 / inner 4 well GroupKFoldのstrict nested scoreでglobal、near `000_050`、
`1000_plus`、worst-wellを確認する。train guard通過後だけsaved-selector inferenceを行い、
current testのcontext 143列すべてでmissing/nonfinite 0を確認する。

## 所見

未実行。Kaggle実測後に更新します。

## Notebook

- `exp245_selector_context_parity_on_exp238_train.ipynb`
- `exp245_selector_context_parity_on_exp238_inference.ipynb`

正の編集対象は対応する Jupytext percent 形式の `.py` です。
