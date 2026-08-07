# exp378_formation_relative_exp226_multisurface_candidate_audit

## 状態

- ルート: pf_beam
- 状態: exp377 Stage 1 scientific FAILにより未実装のまま終了
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: exp226

## 仮説

exp377の相対勾配をexp226物理パスへ戻すと、直接精度または候補集合の多様性を改善できる。

## 変更点

- 6地層単独と固定medianの7候補を生成する。
- Primaryはmedianだけに固定し、7候補から選ばない。
- 固定12候補bankへのadd-only新規性を別gateで測る。

## 検証方針

- Fold: exp226 outer 5-fold
- Group: well
- Stratification: exp226固定
- Leakage Check: exp377 role manifest照合、対象側Formation列0 read

## 実行入口

- 正規notebookはscaffoldのみで未実装。
- exp377 Stage 1 scientific FAILにより実装しない。
- 今回はKaggle package/push/runを行わない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 直接精度と候補新規性を別gateに分けた。

### 悪かった点

- まだ実装・実行していない。

### リスク / 注意

- exp377 median6と個別6面がすべてdirectより悪く、後続3実験へ進まない。

## 次

- 現設計を未実装のまま閉じる。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
