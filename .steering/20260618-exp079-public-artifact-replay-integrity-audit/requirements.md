# 要件

## 依頼

`public_artifact_replay_integrity_audit` を実験として実装する。公開 notebook route の候補を直接 submit する前に、外部生成物依存、static visible CSV、code competition rerun 互換、pairwise submission distance を監査できる notebook と記録を作る。

## 制約

- Route: `pf_beam`
- 最初の対象は `pilkwang/rogii-target-free-tvt-geosteering` とする。
- 次点候補として ridge-sp / SP45 / fle3n / Koolbox 系を同じ監査枠で扱える設定にする。
- 直接 submit から始めない。
- 外部 dataset / kernel / model source は version / hash / offline availability を記録する。
- 公開 LB title だけで採用せず、code competition rerun で生成された `submission.csv` かを確認する。
- Pilkwang notebook の exact-match recovery / guarded overlap override は現設定では無効なので、改善根拠として扱わない。
- 再現性: `docs/06_reproducibility.md` に従い、SHA 記録方針と stochastic 処理の有無を設計に明記する。

## 受け入れ基準

- `experiments/exp079_public_artifact_replay_integrity_audit/` に audit notebook と補助 module がある。
- `config.yaml` に対象 notebook、外部 input slug、branch file、anchor submission path が明示されている。
- Kaggle input の file inventory と SHA を保存できる。
- 候補 submission CSV の sample 互換性、ID 欠損/重複/余剰、予測範囲、SHA を保存できる。
- exp027 / exp073 / exp063 anchor が存在する場合は pairwise distance を保存できる。
- notebook source が取得できる場合は metadata と risk pattern を保存できる。
- 監査結果は JSON / CSV / Markdown で `artifacts/` に保存される。
- deterministic anchor として扱わないことが `config.yaml`、`SESSION_NOTES.md`、`result.md` に明記されている。
