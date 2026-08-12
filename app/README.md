# アプリ

実験確認用に任意で使う Streamlit アプリです。

## 実験ダッシュボード

```bash
task app
```

実験スコア表、`experiment_summary.md`、提出履歴を表示します。

## OOF 分析

```bash
task oof-app
```

`task`が利用できない環境では、同名の`make app` / `make oof-app`を使います。

現在形式の`experiments/*/artifacts/`からOOF CSVファイルを探し、簡易テーブル、数値要約、分布ビュー、欠損値要約を提供します。旧実験との読み取り互換のため、旧形式の`experiments/*/features/`と各実験ディレクトリ直下も検索しますが、新しい生成物の保存先には使用しません。
