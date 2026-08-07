# exp379_formation_transition_mixture_exact_hmm

## 状態

- ルート: pf_beam
- 状態: exp377 Stage 1 scientific FAILにより未実装のまま終了
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: exp209

## 仮説

baseと6地層modeをexact HMM内で混合すれば、坑井・区間に応じて有効な物理参照を選び、固定rate priorより改善できる。

## 変更点

- K16区間内はmode固定、境界だけでmode遷移する。
- 初期確率と境界遷移を事前固定する。
- 16坑井Stage 0後にのみfull CVを検討する。

## 検証方針

- Fold: exp209 outer 5-fold
- Group: well
- Stratification: exp209固定
- Leakage Check: exp378のfold roleとSHAを照合

## 実行入口

- 正規notebookはscaffoldのみで未実装。
- 実装指示後も最初は16坑井Stage 0だけを作る。
- full run、package、pushは今回行わない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 計算量と退化をfull run前に止められる設計にした。

### 悪かった点

- 実装・結果はまだない。

### リスク / 注意

- mode拡張でstate空間とメモリが約7倍方向へ増える。

## 次

- exp377のformation-relative pathが全fold・全6面で悪化したため、現設計を未実装のまま閉じる。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
