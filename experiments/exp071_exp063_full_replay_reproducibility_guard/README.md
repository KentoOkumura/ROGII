# exp071_exp063_full_replay_reproducibility_guard

## 状態概要

- ルート: `ml_model`
- 状態: `discarded`（実装前に中止）
- 作成日: 2026-06-14
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 仮説要約: exp063のfull replayを再現性向け設定で再実行する計画だった。
- 変更点要約: 実装前にCPU-only feature cacheを作るexp072へ方針を移した。
- リスク: 実装・検証・Kaggle実行の証拠はない。
- 次: `exp072_exp063_full_replay_feature_cache`を参照する。

## 正の記録

- 数値: [`metrics.json`](metrics.json)
- 結果と実行証拠: [`result.md`](result.md)
- 作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 実装前の要件と設計: [`docs/legacy/steering/20260614-exp071-exp063-full-replay-reproducibility-guard/`](../../docs/legacy/steering/20260614-exp071-exp063-full-replay-reproducibility-guard/)

## 実行入口

- 学習notebook: `exp071_exp063_full_replay_reproducibility_guard_train.ipynb`
- 推論notebook: `exp071_exp063_full_replay_reproducibility_guard_inference.ipynb`
- この実験は実装前に中止しているため、実行対象にしない。
