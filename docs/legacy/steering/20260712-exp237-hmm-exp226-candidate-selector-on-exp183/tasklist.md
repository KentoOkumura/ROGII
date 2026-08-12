# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering作成。
- exp183を親にexp237を作成。
- 11候補結合、source contract、candidate readout、error-only ranker、fixed Viterbiを実装。
- 静的構文・F821検証を実行。
- Kaggle CPU train v1 を完了（5 boosters、3,051.086 sec）。
- source contract、candidate oracle、CV、bucket、hidden-like、worst-well、path switchを確認。
- global CV は支持したが、near 000_050 と worst-well guard不通過のため inference / submit / rank-slot follow-upを停止。
- fixed Viterbi raw-test artifact生成の実装と静的検証。raw-test cluster/prior OOF-only feature gapはfold-train median fallbackとして明示した。
- Kaggle CPU raw-test inference v2を完了（213.659 sec、14,151 rows / 3 wells）。ID・有限値・selected prediction整合性とprediction / submission SHAを確認。competition submitは0件。
- raw-test long feature 320本のmedian / zero fallbackと、全行`pf_ancc`選択を記録。guard不通過とfeature parity gapのため提出候補にしない。
