# exp378_formation_relative_exp226_multisurface_candidate_audit セッションノート

## 目的

7つのformation-relative物理候補について、直接精度とexp226候補bankへの増分価値を切り分ける。

## 現在の状態

- Route: pf_beam
- 状態: exp377 Stage 1 scientific FAILにより未実装のまま終了
- CV: まだなし
- LB: まだなし

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp378 TITLE=formation-relative-exp226-multisurface-candidate-audit
make new-exp EXP=exp378 SLUG=formation_relative_exp226_multisurface_candidate_audit
```

## 変更点

- 7候補順、median primary、固定12+7 bankを事前固定。
- 35 deterministic candidate runs、0 booster。実行は未着手。

## 再現性メモ

- seed policy: no RNG
- stochastic components: なし
- CPU/GPU runtime: CPU設計、未実行
- Kaggle kernel id / version: なし
- input / feature schema SHA: 実装時にexp377 SHAと照合
- feature content SHA: 未生成
- model manifest / model SHA: 対象外
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun check: 未実行

## 次のアクション

1. 現設計を実装・実行しない。
2. exp377のK / bandwidth / surface救済で本件を再開しない。

## 依存gate結果

2026-07-24、exp377 Kaggle CPU v2はStage 0をPASSし、truth-late Stage 1まで完了した。
median6 path RMSEはdirect `16.100131`から`38.776238 ft`へ悪化し、
rate/path改善foldはいずれも`0/5`、609/773 wellsが悪化した。
個別6 formation pathも全てdirectより悪い。本実験が要求するexp377 Stage 1 PASSは
得られないため、コード、Notebook、35 candidate runsを作らず現設計を閉じる。
