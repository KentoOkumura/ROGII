# exp382_formation_physics_candidate_addonly_on_exp335 セッションノート

## 目的

exp378の7物理候補をleakなしの20特徴へ変換し、exp335へadd-onlyする。

## 現在の状態

- Route: ml_model
- 状態: exp377 Stage 1 scientific FAILによりexp378不成立、未実装のまま終了
- CV: まだなし
- LB: まだなし

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp382 TITLE=formation-physics-candidate-addonly-on-exp335
make new-exp EXP=exp382 SLUG=formation_physics_candidate_addonly_on_exp335
```

## 変更点

- 親exp335、20列、outer5×inner4、15 boosters、0 controlを固定。
- HMM/PF出力は除外。実装・GPU実行なし。

## 再現性メモ

- seed policy: fixed global seed + fixed nested fold manifest
- stochastic components: 物理特徴はなし、LightGBM seed群は実装時固定
- CPU/GPU runtime: feature CPU / train Kaggle T4、未実行
- Kaggle kernel id / version: なし
- input / feature schema SHA: 未生成
- feature content SHA: 未生成
- model manifest / model SHA: 未生成
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun check: 未実行

## 次のアクション

1. 現設計を実装・実行しない。
2. strict-nested特徴生成と15 GPU boostersを開始しない。

## 依存gate結果

2026-07-24、exp377 v2はStage 0をPASSしたが、truth-late Stage 1で
median6 pathがdirectより`22.676107 ft`悪化し、個別6面も全悪化した。
exp378の7候補artifactとnovelty evidenceを作らないことが確定したため、
物理特徴やGPU学習側で救済せず、0 boosterのまま現設計を閉じる。
