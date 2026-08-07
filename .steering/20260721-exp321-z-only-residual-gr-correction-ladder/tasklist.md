# タスクリスト

## 将来の実行・条件付き実装

- なし。Run ABでStage B固定gateをFAILしたため、Stage C、inference、submission、予約案4/5を閉じた。

## 閉鎖

- Stage C実装・実行: Stage Bのbank range / quantization gate FAILにより未実装のまま閉鎖。
- 案4: Stage B PASS条件不成立により未採番・未実装のまま閉鎖。
- 案5: Stage C未到達により未採番・未実装のまま閉鎖。
- shift bank、sigma、threshold、decoderの事後救済、inference、submissionは行わない。

## 完了

- [x] 2026-07-21の追加依頼をStage A/B実装開始の明示承認として記録した。
- [x] Stage A/B用compact self-contained Jupytext train sourceと正規Notebookを実装した。
- [x] anchor/prefix/row/fold/block identity hard guardとtarget-free path/score freezeを実装した。
- [x] Stage A residual structure readout、H256/H512 relative-shape、H512 explained/cap4固定gateを実装した。
- [x] exp280 parityの13 shift、512行block、Gaussian raw-GR emission、stable shuffleと固定gateを実装した。
- [x] fail-closed inference Jupytext source / Notebookを実装し、Stage C・submissionをdisabledに保った。
- [x] 専用10 tests、Jupytext round-trip、構文、ruff F821、strict experiment validationを通した。
- [x] 2026-07-21の追加依頼「実行してください」をKaggle CPU Run ABの明示承認として記録した。
- [x] 1→2→3を単一expの段階gateとして固定した。
- [x] Z-only式、anchor、row/prefix validity、fold、block、shift、GR emission、window補正定数を固定した。
- [x] Run ABとRun Cを分離し、truth-free freeze後だけ採点する境界を固定した。
- [x] Stage A/B/Cのpromotion gate、停止条件、禁止gridを固定した。
- [x] 案4/5を未採番の条件付き別expとして固定し、exp321の範囲外と明記した。
- [x] 実装、Kaggle package/push/run、inference、submissionをdisabledのままにした。
- [x] exp305のKaggle完了を確認した。
- [x] canonical Kaggle CPU kernelへRun AB version 1を一度だけpushし、COMPLETEまで監視した。
- [x] 3,783,989 rows / 773 wells / 5 folds、1 diagnostic / 0 booster / control再実行0を確認した。
- [x] Stage A PASS / Stage B FAILを記録し、Stage Cと予約案4/5を救済なしで閉じた。
- [x] metrics、artifact SHA、result、SESSION_NOTES、experiment summary、strategy backlogを更新した。
