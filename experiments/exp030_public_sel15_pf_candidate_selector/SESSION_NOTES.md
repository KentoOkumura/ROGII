# exp030_public_sel15_pf_candidate_selector セッションノート

## 目的

`exp029` の public sel15 PF/Beam OOF-like artifact を使い、候補選択が same-OOF 過適合ではなく fold 外でも安定するか監査する。

## 現在の状態

- Route: pf_beam
- 状態: 実装中
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp030_public_sel15_pf_candidate_selector
uv run python scripts/new_experiment.py --name exp030_public_sel15_pf_candidate_selector
uv run ruff check experiments/exp030_public_sel15_pf_candidate_selector/candidate_selector_audit.py
uv run python scripts/validate_experiment.py --experiment exp030_public_sel15_pf_candidate_selector
uv run python experiments/exp030_public_sel15_pf_candidate_selector/candidate_selector_audit.py
```

### 予定

```bash
uv run python scripts/update_experiment_summary.py
```

## 変更点

- `candidate_selector_audit.py` を追加。
- fixed candidate、blend、confidence fallback の候補監査を追加。
- original-fold / well-hash の候補選択監査を追加。
- exp026 OOF 候補は exp029 artifact で未接続のため今回から除外。

## 結果

- rows: 1,782,279
- wells: 773
- raw public PF selector: 15.172636
- best same-OOF: `pf090_hold010` 15.089532
- leave-one-original-fold-out candidate selection: 15.141132
- well-hash holdout candidate selection: 15.131490
- leave-one-original-fold-out bucket selection: 15.157679
- well-hash holdout bucket selection: 15.183372
- 判断: fixed conservative blend は支持あり。confidence fallback / bucket hard selector は不安定。

## 次のアクション

1. `experiment_summary.md` と `backlog/KAGGLE_DIRECTION.md` を更新する。
2. `pf090_hold010` の inference 移植可否を確認する。
