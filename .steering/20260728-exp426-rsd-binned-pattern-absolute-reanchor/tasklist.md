# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- Stage B / C0 / C1、inference、submissionはStage A technical FAILにより
  fail-closeした。parameter rescueは行わない。

## 完了

- Stage A compact self-contained Jupytext train sourceと対応するnotebook候補を
  実装した。
- RSD-binned Pearson、raw pointwise Pearson、exp280互換raw Gaussian、
  stable permutation、truth-late discrete oracle / replay / gateを実装した。
- target-free freeze、fixed probe rerun、logical / decompressed SHA、
  technical FAIL時のtruth未読fail-closeを実装した。
- Stage A contract testsを追加した。
- compact候補を正規train Notebookへ採用し、Kaggle private CPU version 1
  （id_no `128930757`）を実行した。
- 3,783,989 rows / 773 wells / 7,787 blocks、101,231 score rowsを生成し、
  runtime `164.719113 sec`、peak RSS `0.803265 GB`、fixed-probe parityを確認した。
- supported blocks `25.593939% < 95%`、supported wells
  `89.262613% < 98%`によりtechnical FAILとし、truth / hidden-like role未読の
  ままscientific評価をskipしてterminal closeした。
- 実行結果、gate、read ledger、content SHA、終端判断をconfig、metrics、
  result、README、SESSION_NOTESへ記録した。
- Wu et al. (2019) のRSD projection、0.5 ft bin mean、correlation、
  physical prior、SAMC、不確実性の適用境界を確認した。
- exp408のHMM translation-gauge lockとcurrent-emission非支配を確認した。
- exp226のsegment offset oracle、persistent SSE、suffix距離依存を確認した。
- exp410のparticle support不足とseed / particle multiplicityを確認した。
- exp280 / exp360のraw Gaussian / ZNCC negative evidenceを確認した。
- Stage A / B / Cの単一candidate contract、gate、実行量、fail-closeを固定した。
- Stage Cのaugmented target、uniform control、guided defensive proposal、
  `p/q<=2`を式で固定した。
- `docs/06_reproducibility.md`を読み再現性設計へ反映した。
- steeringとdesign-only実験scaffoldを作成した。
- 作成中のexp番号競合を検出し、既存exp425を保持して本実験をexp426へ繰り上げた。
- `KAGGLE_DIRECTION.md`へP3候補として追加した。
- `experiment_summary.md`へdesign-only実験を反映した。
- `make validate-exp` strict PASS、YAML / JSON parse PASSを確認した。
- experiment docs reviewでcore evidence categoriesが揃っていることを確認した。
