# 調査サマリー

このファイルは、複数の完了レポートに共通する、変わりにくい横断結論だけを示します。個別の証拠と最新の完了レポートは[`README.md`](README.md)の生成索引から探してください。現在の比較基準、優先順位、未解決事項は[`KAGGLE_DIRECTION.md`](../../KAGGLE_DIRECTION.md)を正とします。

## 横断結論

- 評価はwell単位の分割で行い、validation wellの`TVT_input`がNaNの行だけを対象にする。公開3 wellsの再現やPublic LBだけではhidden testへの一般化を判断しない。
- train-only formation columnsはhidden testで利用できる前提にせず、推論時に生成可能な情報だけで比較する。
- typewell GRとhorizontal GRの対応、軌道、既知区間の`TVT_input`は主要な情報源だが、GR matchingや既知区間からの直接補正はwell-tailを悪化させることがある。単独の置換だけでなく、候補、尤度、confidence、risk特徴としての利用範囲を検証する。
- PF、Beam、HMM、公開Notebook由来の予測は、候補生成や不確実性の情報源として有用な場合がある。一方、平均CVの改善だけでは採用せず、fold、hidden-like条件、by-well tail、hidden testでの再生成可能性を併せて確認する。
- 複数候補の選択や融合は、同じOOF上での救済探索を避け、outer-foldで学習と評価を分離する。pooled RMSEだけでなくfold一貫性とworst-well側の悪化を停止条件に含める。
- 公開解法の再現値は、作者報告値や公開Notebookの説明と区別する。実装契約、省略した処理、Kaggle実行で得たCV・Public LB・Private LBを分けて記録する。

## 更新方針

新しい完了レポートが既存の横断結論を変えた場合だけこのファイルを更新します。実験ごとの結果や時系列、現在の候補、次の実験案はここへ複製しません。
