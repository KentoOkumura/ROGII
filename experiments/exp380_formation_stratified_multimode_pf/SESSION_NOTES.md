# exp380_formation_stratified_multimode_pf セッションノート

## 目的

formation-relative物理候補を粒子modeとして保持するstratified PFを検証する。

## 現在の状態

- Route: pf_beam
- 状態: exp377 Stage 1 scientific FAILにより未実装のまま終了
- CV: まだなし
- LB: まだなし

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp380 TITLE=formation-stratified-multimode-pf
make new-exp EXP=exp380 SLUG=formation_stratified_multimode_pf
```

## 変更点

- 粒子600、7 mode、最低配分、stable SHA seedを固定。
- Stage 0=773、Stage 1=3,092 runs。実行なし。

## 再現性メモ

- seed policy: stable SHA per split/fold/well/mode/seed
- stochastic components: 初期化、noise、resampling、adaptive配分
- CPU/GPU runtime: CPU設計、未実行
- Kaggle kernel id / version: なし
- input / feature schema SHA: 未生成
- feature content SHA: 未生成
- model manifest / model SHA: PF manifest未生成
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun check: 単独/並列/順序変更一致をStage 0前に確認予定

## 次のアクション

1. 現設計を実装・実行しない。
2. seed 0の773 runsとmean4の3,092 runsを開始しない。

## 依存gate結果

2026-07-24、exp377 v2はStage 0をPASSしたが、truth-late Stage 1で
median6 pathがdirectより`22.676107 ft`悪化し、個別6面も全悪化した。
exp378も未実装のまま終了するため、PF側の粒子配分でscientific negativeを救済せず閉じる。
