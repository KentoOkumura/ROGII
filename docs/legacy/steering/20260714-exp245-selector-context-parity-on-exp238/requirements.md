# 要件

## 依頼

exp238 selector の train/test feature parity を修正する。visible test で全行 `NaN` だった
45 context 列を、hidden test の well 数に依存しない再現可能な契約へ変更する。
この修正を通過した selector を、次段の候補 path 直接採用実験の入力にする。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- outer 5 / inner 4 well GroupKFold の strict nested stacking を維持し、outer-valid well の正解 TVT を selector 学習へ流さない。
- exp109/114 OOF-only `copcf_*` は test-equivalent generator がないため、train と test の両方から除外する。
- exp226 の `geop` / `gr_delta` は current test 上で行単位に再生成し、学習時と同じ4診断列を作る。
- exp245 は selector 監査までとし、final LightGBM、direct replacement、提出は実行しない。

## 受け入れ基準

- selector context は学習・推論で同一schemaとなり、current test の missing context 列数が0である。
- context 全列と11候補値について非有限値が0である。
- selectorは1 config × outer 5 × inner 4 = 20 CPU boostersだけを学習し、親/control/final LightGBMを再学習しない。
- global、near `000_050`、`1000_plus`、worst-wellをexp238と同じguardで保存する。
- outer/inner fold、context schema、saved model、nested scoreのmanifestとSHAを保存する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
