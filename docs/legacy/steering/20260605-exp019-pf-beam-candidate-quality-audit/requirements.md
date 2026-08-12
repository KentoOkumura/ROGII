# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ先頭 `pf_beam_candidate_quality_audit` を
実装する。`exp015` の PF/beam add-only features が悪化した原因を、追加学習や
提出なしで診断する。

## 制約

- 親実験は `exp015_public_pf_beam_scale_selector_features`。
- `exp013` raw `lightgbm_no_gr` OOF を比較対象にする。
- train-only formation columns は使わない。
- Kaggle Notebook の再実行、再学習、提出はしない。
- この監査自体の実行場所は Kaggle Notebook とし、ローカルで全件監査を実行しない。
- PF/beam candidate paths は train CSV と typewell GR から deterministic に再計算する。
- 同一 OOF 上の direct candidate score は診断値として扱い、提出候補にはしない。

## 受け入れ基準

- `exp019_pf_beam_candidate_quality_audit` が作成されている。
- PF/beam direct candidate、scale、confidence、GR gap、距離 bucket、well 条件別の集計 artifact が出る。
- `exp015` feature-model の hurt/help well が raw control と比較できる。
- `metrics.json` と `result.md` に raw anchor、best direct PF candidate、採用判断が記録される。
- `KAGGLE_DIRECTION.md` の実装済み backlog が整理され、次候補の優先度が更新される。
