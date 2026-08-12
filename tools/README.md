# ツール

`tools/` は、このリポジトリが実装・配布するスクリプトではなく、作業時に利用する外部ツールの clone やローカル配置に使います。外部ツール本体は Git で追跡しません。

このリポジトリが管理する自動化、検証、記録更新、外部ツールの起動ラッパーは `scripts/` に置き、`Taskfile.yml` または `Makefile` から呼び出します。

## ROGII viewer

Kaggle discussion 700424 で共有された Tom さんの PySide viewer は `tools/rogii-viewer/` に配置しています。このディレクトリは外部 clone として `.gitignore` しています。

```bash
task viewer
```

既定では `data/raw` を dataset folder として開きます。別のデータ置き場を使う場合:

```bash
task viewer VIEWER_DATA=/path/to/rogii-wellbore-geology-prediction
```

smoke確認は`task viewer-smoke`を使います。`task`が利用できない環境では、同名の`make viewer` / `make viewer-smoke`を使います。
