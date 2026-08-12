# タスクリスト

## 未着手

- 別承認後、Jupytext percent形式のcompact self-contained inference sourceを実装する。
- exp374 compact sourceから必要なexact-HMM関数だけを抽出し、source/config SHAを照合する。
- Student-t式、df、grid、rate、sigma、補完、posterior meanのcontract testを作る。
- dynamic sample/ID/nonempty-well、finite、posterior、fallback 0、実行量、SHA guardを実装する。
- Gaussian/Huber/PF/Beam/controlが実行されないことをtestする。
- 正規inference Notebook採用前にJupytext test、py_compile、Ruff F821、
  専用test、`task validate-exp`を通す。
- Kaggle push前にmetadataとbootstrap内configの整合を確認する。
- 別承認後、Kaggle CPU inferenceを実行し、outputを取得してsubmit-checkする。
- output取得後にinput/content SHA、prediction SHA、submission SHA、
  kernel version、HMM実行量を記録する。
- さらに別のcompetition submission承認後、凍結順3番目として提出・監視する。
- Public LB確定後、exp434 exact HMMとの差をresult/metrics/summaryへ記録する。

## 進行中

- なし

## ブロック中

- 実装、Kaggle実行、competition submissionはいずれも未承認。

## 完了

- exp454を採番し、標準experiment scaffoldを作成した。
- 候補をfixed `df=4.0` Student-t exact HMMの1本に固定した。
- evaluation/inference Notebookを1本とする契約を固定した。
- HMM状態空間、emission、hidden cardinality、LB解釈、禁止事項を固定した。
- 再現性設計を`design.md`と`config.yaml`へ記録した。
