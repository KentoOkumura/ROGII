# 共通テスト

このディレクトリには、複数実験で共有するコードやリポジトリ全体の契約を検証するテストを置きます。

単一実験だけに属するテストは、対応する`experiments/<exp>/tests/`へ置きます。ルートで`task test`または`make test`を実行すると、Notebook依存を含む環境で両方の場所が収集・実行されます。

実験固有テストを収集せず、このディレクトリだけを確認する場合は`task test-common`または`make test-common`を使います。同梱skillの構造・任意のUI metadata・Pythonコードの静的検査は`task check-skills`または`make check-skills`で行います。
