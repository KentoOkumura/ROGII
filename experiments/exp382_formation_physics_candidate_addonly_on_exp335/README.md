# exp382_formation_physics_candidate_addonly_on_exp335

## 状態

- ルート: ml_model
- 状態: exp377 Stage 1 scientific FAILによりexp378不成立、未実装のまま終了
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: exp335

## 仮説

formation-relative物理候補をstrict-nested特徴として加えると、exp335のML予測へleakなしの地層構造信号を追加できる。

## 変更点

- exp335の370特徴へ固定20列だけを追加する。
- outer5×inner4、計25 partitionで物理候補を再生成する。
- 1 variant×3 config×5 fold=15 booster、control再学習0。

## 検証方針

- Fold: exp335 outer 5-fold、各outer内inner 4-fold
- Group: well
- Stratification: exp335固定
- Leakage Check: saved 5-fold OOF流用禁止、role別donor/read audit

## 実行入口

- 正規notebookはscaffoldのみで未実装。
- exp377 Stage 1 scientific FAILによりexp378が成立しないため実装しない。
- 今回はGPU学習、package、push、推論、提出を行わない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- fold mismatch leakageを設計段階で排除した。

### 悪かった点

- 実装・CV結果はまだない。

### リスク / 注意

- nested生成コストとGPU 15 boosterが必要。実行前に再承認する。

## 次

- 現設計を未実装・0 boosterのまま閉じる。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
