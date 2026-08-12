---
name: kaggle-notebook-fetch
description: "Kaggle CLI を使って、コンペの上位公開ノートブックをメタデータ付きでローカルに保存する。Kaggle 調査、公開ノートブックのキャッチアップ、高 vote ノートブックの取得、ローカルアーカイブの更新、上位解法の議論準備に使う。"
---

# Kaggle ノートブック取得

同梱スクリプトでコンペの公開ノートブックを一覧し、指定した sort 順の kernel をメタデータ付きで取得する。

## 手順

1. `project.yml`の`competition.slug`が対象コンペであることを確認する。
2. ノートブックを取得する。

```bash
task fetch-kaggle-notebooks EXTRA_ARGS="--limit 20"
```

`task` がない環境では、同名の `make fetch-kaggle-notebooks EXTRA_ARGS="--limit 20"` を使う。`project.yml`とは別のコンペを取得する場合だけ`COMPETITION=other-competition`で上書きする。

3. 既定の保存先は `docs/notebooks/<competition>/`。
4. 再実行しても安全。既存フォルダは、`kernel-metadata.json`の`code_file`が存在し、非空の場合だけスキップされる。不完全な取得は自動で再取得し、完全な取得を更新するときだけ`--force`を使う。
5. vote 順以外を見る場合は `--sort-by scoreAscending`、`--sort-by dateRun` などを指定する。

## メモ

- スクリプトは `kaggle kernels list --competition ... --sort-by SORT -v` を使う。既定は `voteCount`。
- ダウンロードは `kaggle kernels pull OWNER/SLUG -m` で行い、`kernel-metadata.json` を保持する。
- API が一時失敗する場合は `--retries` を増やす。取得できないkernelがあっても残りを処理するが、metadataまたは非空のcode fileを取得できない項目を表示し、最後は非0で終了する。
- 認証は Kaggle CLI v2.2.0+ なら `uv run kaggle auth login`、`KAGGLE_API_TOKEN` / `~/.kaggle/access_token`、またはlegacy `KAGGLE_USERNAME` / `KAGGLE_KEY`・`~/.kaggle/kaggle.json`を使う。
- 大量に取得する前に `--dry-run` を使う。
- Kaggle CLI の出力形式が変わった場合は、推測で進めず、生成された CSV や一覧出力を確認してスクリプトを修正する。

取得後は、保存した参照先を要約し、最初に読む価値が高いノートブックを提案する。
