# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steeringでexp304の入力面、3 denoiser、readout、promotion gate、失敗時の打ち切りを固定した。
- exp304の実行量を`0 models / 0 boosters / 0 HMM / 0 PF / 0 Beam`として固定した。
- 案2〜4を別expの予約契約として設計し、依存関係と禁止事項を固定した。
- 2026-07-20: ユーザーからexp304の実装承認を受けた。
- compact self-contained Jupytext train/inference sourceを実装した。
- raw/robust RTS/SWT/L1 trend、target-free score freeze、late truth join、scope/fold readout、
  technical/quality gate、全expected artifact保存を実装した。
- synthetic unit test、構文、ruff、Jupytext変換test、experiment validationを完了した。
- Kaggle実行、HMM/PF/Beam、inference、submissionは今回のscope外として維持した。
- 2026-07-20: ユーザーから正規train採用とprivate Kaggle CPU v1の実行承認を受けた。
- compact trainを正規train Notebookへ採用し、`--no-src --strict` package preflightを完了した。
- Kaggle private CPU version 1（id_no `128011752`）を完了した。
- actual input/scientific contract/denoised GR/target-free score SHAとkernel versionを記録した。
- raw/SWTはtechnical PASS、RTS/L1はtechnical FAILと判定し、部分scoreをquality評価から除外した。
- `swt_db4_l3`がMRR/top3を5/5 foldsで改善して全quality gateを通過し、唯一のselected denoiserに確定した。
- exp304を完了し、案2を別exp候補へ開いた。SWT選択のため案3は閉じ、案4は案2 PASS待ちとした。
