# タスクリスト

## TODO

- Kaggle に push する前に competition policy / leakage 解釈を確認する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `.steering` に要件と設計を記録した。
- `config.yaml` に既知 public sample wells と assert policy を明記した。
- well id 抽出と assert probe の補助モジュールを実装した。
- train notebook を public sample sanity 用に更新した。
- inference notebook を hidden overlap assert probe と submission 作成用に更新した。
- ローカル関数で public sample の既知 overlap sanity を確認した。
- `make validate-exp EXP=exp064_train_test_well_id_assert_probe` を実行した。
- `make prepare-kaggle-notebooks EXP=exp064_train_test_well_id_assert_probe EXTRA_ARGS="--notebook inference --run-on-push --strict"` を実行した。
- `make record-exp` で `experiment_summary.md` に未実行実装として記録した。
