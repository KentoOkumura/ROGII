# exp510_exp413_exact_public_preoverride_hedge

## 概要

- 仮説要約: 公開解法のwell固有処理前の予測をexp413へ固定比率で加えることで、公開データ分布への別系統の予測を組み合わせる。
- 変更点要約: 公開testに固定されたexp413予測を使わず、実行時のsampleに対してexp413を再生成してから、設定済みの固定式で予測を合成する。
- リスク: 対応するhonest OOFとPrivate LBがなく、Public LBだけでは汎化性能を判断できない。
- 次: 正規inference notebookへの採用と科学的な採否をユーザーが判断するまで、追加提出やweight変更を行わない。

## 正の記録

- 数値、実験status、kernel情報、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行コマンド、途中経過、失敗と修正: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 実装前の要件と設計: [`.steering/20260804-exp510-exp413-exact-public-preoverride-hedge`](../../.steering/20260804-exp510-exp413-exact-public-preoverride-hedge/)

## 実行入口

- 学習 notebook: `exp510_exp413_exact_public_preoverride_hedge_train.ipynb`（この推論実験では学習しない）
- 正規推論 notebook: `exp510_exp413_exact_public_preoverride_hedge_inference.ipynb`（未採用のplaceholder）
- 検証済み候補: `exp510_exp413_exact_public_preoverride_hedge_compact_selfcontained_inference.py` / `.ipynb`
- hidden test対応の生成スクリプト: `prepare_exp510_hidden_safe_runtime.py`
- 実験固有テスト: `tests/test_exp510_exp413_exact_public_preoverride_hedge.py`
- Kaggleでの実行と再実行手順は`SESSION_NOTES.md`を参照し、追加実行・提出は新たなユーザー承認後に行う。
