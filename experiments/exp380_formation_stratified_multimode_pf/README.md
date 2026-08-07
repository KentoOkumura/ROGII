# exp380_formation_stratified_multimode_pf

## 状態

- ルート: pf_beam
- 状態: exp377 Stage 1 scientific FAILにより未実装のまま終了
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: exp271（PF実装: exp072）

## 仮説

formation別の粒子枠を保証すれば、通常のresamplingで早期消失する地層仮説を保持し、PFの多峰性を有効利用できる。

## 変更点

- 600粒子をbase 300、各formation 50に層別化する。
- resample後もbase 150、各formation 25を最低保持する。
- seed 0とmean4を段階分離する。

## 検証方針

- Fold: exp271 outer 5-fold
- Group: well
- Stratification: exp271固定
- Leakage Check: exp378 role manifest/SHA照合

## 実行入口

- 正規notebookはscaffoldのみで未実装。
- 指示後もまずseed 0 Stage 0だけを実装する。
- 今回はPF実行、package、pushを行わない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- stochastic PFのseedと並列再現性を設計時点で固定した。

### 悪かった点

- 実装・結果はまだない。

### リスク / 注意

- 4 seedは3,092 well-seed runsとなるため別承認が必要。

## 次

- exp377のformation-relative pathが全fold・全6面で悪化したため、現設計を未実装のまま閉じる。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
