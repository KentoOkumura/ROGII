# 要件

## 依頼

exp383/384が作る複数の物理vector-drift pathを、対象坑井で観測可能なhorizontal GRと
typewell GRの整合度で周辺化する。最終予測はML回帰器ではなく物理path posteriorとし、
物理モデル単独LB 6.5ロードマップのP2とする。

今回はbacklog、steering、実験ディレクトリと設計だけを確定する。
実装、正規Notebook採用、Kaggle package/push/run、推論、提出は行わない。

## 制約

- Route: `pf_beam`
- 親: `exp384_fault_aware_piecewise_stratigraphic_vector_field`
- exp383とexp384のStage 0/1が全PASSし、別のユーザー承認があるまで実装しない。
- exp384のsmooth-base path、最大8 component paths、posterior/uncertaintyを保存SHAで固定する。
- candidate path値、formation/vector field、prefix calibration、exp226 fallbackを変更しない。
- 対象horizontalの`GR`と対象typewellの`TVT/GR`は推論時に利用可能な観測として使う。
- target suffixの正解TVT、生Formation、error、oracle pathはlikelihoodやstate選択に使わない。
- GR likelihoodのwindow、stride、Student-t df、sigma clip、transitionを同一OOFでgridしない。
- candidate best-ofやhard top1ではなく、exact forward-backward posterior平均をprimaryにする。
- GR欠損/低coverage位置は親exp384へexact fallbackする。
- LightGBM/CatBoost/XGBoost/NN、particle sampling、seed baggingは使わない。
- 公開3 sample wellsで設定を選ばない。

## 受け入れ基準

- steering、実験scaffold、config、README、SESSION_NOTES、result、metricsに
  同じconditional design-only契約がある。
- `KAGGLE_DIRECTION.md`でexp383/384 PASS条件付きP2として記録される。
- Stage 0でcandidate bank、typewell interpolation、GR window score、transition、
  posteriorをtarget truth前にfreezeする。
- candidate countはbase 1 + component最大8の最大9で固定する。
- eligible GR window率`>=0.25`、eligible well率`>=0.50`、posterior finite/normalizationを満たす。
- known-prefix heldout readoutで親より改善し、real GRがcircular-shift controlより識別力を持つ。
- Stage 1でexp384比`0.50 ft`以上、4/5 folds、1000+/hidden-like改善を要求する。
- GR欠損bucketとineligible rowsは親非悪化かつexact fallback parityを満たす。
- full 773-well exact decoder runはStage 0 PASS後の別承認を必要とする。
