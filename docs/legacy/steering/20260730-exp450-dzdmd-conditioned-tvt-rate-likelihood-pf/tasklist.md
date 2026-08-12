# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Stage 0Bが事前固定16 gate中10 PASS・6 FAILのため、Stage 1、再実行、
  parameter/grid救済、inference、submissionへ進まない。

## 完了

- 学習型`mu=beta*dZ/dMD+intercept`をユーザー確認で選択した。
- exp072 `PF_Z`、exp404/417 likelihood-PF、exp446 negative mechanismを確認した。
- 科学差分、fallback、exact-coordinate parity、段階、実行量、全AND gate、
  禁止事項、再現性とtruth-late契約を固定した。
- steeringを実験ディレクトリより先に作成した。
- 2026-07-30のユーザー依頼でcompact self-contained train候補と
  inference fail-closed guard候補をJupytext percent形式で実装した。
- prefix OLS/fallback/tail20、residual-AR、exact beta=-1 paired parity、
  first-row、resampling/roughening draw parity、truth-late、SHA、
  Stage 0A/0B gateの専用testを作成した。
- 正規Notebookを上書きせず、compact `.ipynb`候補へ変換した。
- 2026-07-30にユーザーから実行承認を受領し、正規train Notebookを採用した。
- push前にscientific variant 1、Stage 0A 24 PF well-runs、
  Stage 0B 32 PF well-runs、control rerun 0、model/booster/GPU 0を再確認した。
- Kaggle private CPU version 1（id_no `129167787`）をpushし、`COMPLETE`まで
  監視した。
- Stage 0Aの24 PF well-runsを実行し、exact-coordinate parity FAILを記録した。
- fail-closedによりStage 0Bを開始しなかった。
- scientific contract、parity report、paired prediction raw/decompressed SHAを
  記録した。
- ユーザーからtemperature-5集約予測の微小丸め差を許容してStage 0Bへ進む
  明示承認を受領した。
- version 2は改訂Stage 0AをPASSし、Stage 0B candidate 32 wellsを生成した。
- version 2は保存exp404 source logical SHAの計算方式不一致でERRORとなった。
- exp404と同じtyped logical SHA関数・dtype正規化を移植し、親関数との一致testを
  PASSした。
- typed SHA修正版を同一canonical kernelのversion 3としてpushし、
  `COMPLETE`まで監視した。
- version 3は改訂Stage 0AをPASSし、Stage 0Bの32 candidate wellsを完走した。
- Stage 0Bの全16 gate、56 PF well-runs、7,168 seed-well、
  3,584,000 particle starts、主要artifact SHAを記録した。
- Stage 0Bはprefix backtestをPASSした一方、under-response、episode、
  fold再現性、matched-control安全性の6 gateをFAILしたため、
  `stage0b_mechanism_failed_closed`として終了した。
