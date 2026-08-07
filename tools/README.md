# ツール

実験に依存しない、リポジトリ全体のユーティリティです。

例:

- Kaggle データセットのアップロード補助。
- 提出ファイルの検証。
- ランタイム監視。
- レポート生成。

## ROGII viewer

Kaggle discussion 700424 で共有された Tom さんの PySide viewer は `tools/rogii-viewer/` に配置しています。このディレクトリは外部 clone として `.gitignore` しています。

```bash
make viewer
```

既定では `data/raw` を dataset folder として開きます。別のデータ置き場を使う場合:

```bash
make viewer VIEWER_DATA=/path/to/rogii-wellbore-geology-prediction
```

`task` が入っている環境では `task viewer` / `task viewer-smoke` でも同じ launcher を使えます。
