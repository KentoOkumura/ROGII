---
name: kaggle-notebook-fetch
description: "Kaggle CLI を使って、コンペの上位公開ノートブックをメタデータ付きでローカルに保存する。Kaggle 調査、公開ノートブックのキャッチアップ、高 vote ノートブックの取得、ローカルアーカイブの更新、上位解法の議論準備に使う。"
---

# Kaggle ノートブック取得

同梱スクリプトでコンペの公開ノートブックを一覧し、指定した sort 順の kernel をメタデータ付きで取得する。

## 手順

1. コンペの slug を特定する。
2. ノートブックを取得する。

```bash
python .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition COMPETITION --limit 20
```

3. 既定の保存先は `docs/notebooks/<competition>/`。
4. 再実行しても安全。既存のノートブックフォルダは、`--force` を付けない限りスキップされる。
5. vote 順以外を見る場合は `--sort-by scoreAscending`、`--sort-by dateRun` などを指定する。

## メモ

- スクリプトは `kaggle kernels list --competition ... --sort-by SORT -v` を使う。既定は `voteCount`。
- ダウンロードは `kaggle kernels pull OWNER/SLUG -m` で行い、`kernel-metadata.json` を保持する。
- API が一時失敗する場合は `--retries` を増やす。取得できない kernel があってもスクリプトは失敗を記録して次の kernel に進む。
- 認証は Kaggle CLI v2.2.0+ なら `kaggle auth login`、または `KAGGLE_API_TOKEN` / `~/.kaggle/access_token` を使う。
- 大量に取得する前に `--dry-run` を使う。
- Kaggle CLI の出力形式が変わった場合は、推測で進めず、生成された CSV や一覧出力を確認してスクリプトを修正する。

取得後は、保存した参照先を要約し、最初に読む価値が高いノートブックを提案する。
