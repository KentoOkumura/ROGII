# アプリ

実験確認用に任意で使う Streamlit アプリです。

## 実験ダッシュボード

```bash
streamlit run app/streamlit_app.py
```

実験スコア表、`experiment_summary.md`、提出履歴を表示します。

## OOF 分析

```bash
streamlit run app/oof_analysis_app.py
```

`experiments/*/artifacts/`、`experiments/*/features/`、各実験ディレクトリ直下から OOF CSV ファイルを探し、簡易テーブル、数値要約、分布ビュー、欠損値要約を提供します。
