# exp515 3位チームの公開説明を基にした3種類のHMM

## 状態

実装・静的検証中。Kaggle OOF、inference、late submissionは未実行。

## 仮説

Kaggle discussion 733319で公開された3位解法のHMM部分を基に実装する。元コードは非公開であり、状態の刻み方、遷移値、self-referenceの重み、Local-DTWの詳細は推定または近似している。このため、元チームのHMMの忠実な再現ではない。
原チームのコードと未公開hyperparameterは取得できないため、source parityではない。

公開された4要素のhidden state、sibling/self reference、prefix-in-model、Student-t(df=1)、exact forward-backward、3 family固定混合、物理projectionを保持する。未公開値は実行前に`config.yaml`へ固定し、OOFやLBを見た救済調整を行わない。

## 検証方針

well GroupKFoldのvalidation fold全体をsibling atlasから除外し、target-free predictionをSHA凍結した後だけ真のTVTで評価する。hidden inferenceはruntimeのtrain/test/sample submissionを動的列挙し、sample IDと1対1に整列する。

## 所見

数値所見はまだない。原チームのOOF 5.9703、Public 6.207、Private 6.229は外部参照値であり、exp515の実績ではない。

## 次のアクション

Kaggle CPU train/inferenceのtechnical gateを通し、固定版を1回だけlate submitする。

この実験はコンペ終了後に行う。提出前の検証を通った固定版を1回だけ提出し、Notebook titleとsubmission messageに`LATE SUBMIT`を明記する。
