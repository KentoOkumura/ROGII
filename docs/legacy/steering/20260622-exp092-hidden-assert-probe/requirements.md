# 要件

## 依頼

`exp092_u_projection_correction_disagreement_fullrun` の実験目的を、通常 notebook の visible-test guard ではなく、Code Competition の submission rerun で hidden test 条件を pass/fail だけで間接観測できる opt-in assert probe に合わせる。

通常 Kaggle notebook 実行で読める `test` は exposed sample / visible test であり、hidden LB test は code submission rerun 時に差し替えられる。したがって、hidden 側で見たい事象は inference notebook 内に assertion として仕込み、submission が通ったか落ちたかだけを信号として扱う。

## 制約

- Route: `ml_model`
- 既存 `exp092_u_projection_correction_disagreement_fullrun` の follow-up として扱い、duplicate exp フォルダは作らない。
- probe は opt-in とし、デフォルトでは無効にする。
- probe 有効時も normal notebook の visible test では hidden assertion を skip できるようにする。
- hidden test 由来の行数、well 数、集計値、予測統計を assertion error に出さない。失敗時は check 名だけを出す。
- probe は submission.csv を改変しない。検査は prediction / feature / sample-id coverage の後、submission 書き出し前後の境界で実行する。
- 再現性: `docs/06_reproducibility.md` に従い、hidden code-submit rerun での SHA と pass/fail 記録方針を設計に明記する。

## 受け入れ基準

- `run_saved_model_inference()` が `hidden_assert_probe` config を受け取り、デフォルト無効で従来 inference と同じ挙動を保つ。
- exp092 inference notebook が `inference.hidden_assert_probe` を読み、実行関数へ渡す。
- probe 有効時、visible-test signature を検出して skip できる。
- hidden context では、設定済み assert 条件に違反した場合に `AssertionError` を投げ、失敗 check 名だけを表示する。
- config / result / SESSION_NOTES / KAGGLE_DIRECTION / experiment_summary に、visible guard ではなく hidden assert probe が実験目的であることを記録する。
- static validation、notebook JSON validation、self-test 相当の関数チェックを通す。
