# exp515 3位チームの公開説明を基にした3種類のHMM 結果

## 状態

実装・静的検証・Kaggle package生成まで完了した。数値結果はまだない。

Kaggle UI Active SessionsのCPU/GPU `active / limit`をこの実行環境から取得できないため、必須session guardに従ってpushを保留している。

## 解釈上の注意

3位チームの公開codeは取得できなかった。公開説明を基にしているが、状態の刻み方と遷移値を推定し、Local-DTWを3つの伸縮率で近似した実験である。元チームのHMMの忠実な再現とは扱わない。
原チームのOOF `5.9703`、Public `6.207`、Private `6.229`は参照値であり、exp515の実績ではない。

この実験の提出はコンペ終了後の`LATE SUBMIT`であり、競技中のmodel selectionとは分けて記録する。
