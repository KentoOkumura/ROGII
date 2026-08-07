# exp285_exp226_prefix_masked_offset_predictability_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU version 2完了、scientific guard FAIL、branch closed
- CV: なし（train-side predictability readout）
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-19
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

exp226 geometry-only `tvt_geop`の局所形状誤差が低周波で持続するなら、known prefix末尾640行を
pseudo suffixとして隠して測ったoffset median / slope / block driftは、official evaluation suffixの
exp226 residualをfold-stableに予測できる。

## 固定設計

- exp226保存5 folds、fold外donor field、保存fold kappaを使用する。
- pseudo cutはknown prefix末尾640行を隠す1点だけ。cut以前のvisible rowは最低512行。
- pseudo replayはtargetの`X/Y/Z/MD`とvisible anchorだけを使い、geometry-only pathをwell末尾まで作る。
- pseudo path freeze後にmasked `TVT_input`を戻し、5 x 128行blockから3 summaryを作る。
- prefix summary freeze後にだけofficial suffix true TVTを結合する。
- primary guardはpooled/fold別Spearman、sign balanced accuracy、256 fold内permutationで判定する。
- 1 variant / LightGBM 0 config / trained fold 0 / booster 0 / HMM・PF regeneration 0。

詳細は`.steering/20260719-exp285-exp226-prefix-masked-offset-predictability-readout/`と
`config.yaml`を正とする。

## 変更点

exp226のofficial OOFをそのまま再評価するのではなく、validation wellのknown prefixだけを短縮して
geometry-only pathを再生する。GR correctionやU projectionは使わず、prefix内で観測可能なresidual summaryと
official suffix target summaryをfreeze境界の後で比較する。

## 検証方針

- Fold / Group: exp226保存5 folds / `well_id`。
- Primary: pseudo offset median対official full-suffix offset medianのwell単位Spearman。
- Safety: fold別相関、sign balanced accuracy、256回fold内permutation、near / 1000+ / hidden-like。
- Leakage check: validation well全体のdonor除外、masked `TVT_input`とofficial true TVTの段階別access countを0に固定。
- Promotion: technical、primary、supporting、scope guardの全PASSが必要。

## 実装状態

別名compact self-contained Jupytext train / inferenceを実装した。trainは9章・20 cellsで、fold-safe
exp226 replay、pseudo mask、geometry/prefix freeze、official target、correlation/permutation/guard、
生成物保存までを展開する。inferenceは4章・10 cellsのfail-closed実装である。

実行承認後、compact trainを正規train notebookへ採用した。version 1のraw `id`列契約不一致を修正後、
version 2が766 wellsを完走した。正規inference notebookはtemplate stubのまま維持し、ローカルnotebook
実行、current-test生成、推論、提出は行っていない。

## 実行入口

- 正規train notebook: `exp285_exp226_prefix_masked_offset_predictability_readout_train.ipynb`
- 推論notebook stub: `exp285_exp226_prefix_masked_offset_predictability_readout_inference.ipynb`
- compact train: `exp285_exp226_prefix_masked_offset_predictability_readout_compact_selfcontained_train.ipynb`
- compact inference: `exp285_exp226_prefix_masked_offset_predictability_readout_compact_selfcontained_inference.ipynb`
- Kaggle kernel: `kentookumura/exp285-prefix-masked-offset-readout-train` version 2。

## 生成物

pseudo geometry、prefix summary、official target summary、overall/fold/scope/permutation/by-well metrics、
input/contract manifestをKaggle outputへ保存した。ローカル監査先は
`/tmp/kaggle-output/exp285_exp226_prefix_masked_offset_predictability_readout/train_v2`。

## 判定

technical guardは全PASSしたがprimary / supporting / scope guardはFAILした。parameter rescueや
prefix-calibrated correction、exp281 blend/selectorへ進まずbranchを閉じる。

## 所見

full-suffix primary Spearmanは`-0.004135`で予測性がない。H256/nearは約`0.187/0.190`の弱い正相関だが、
H640 `0.131`からfull suffixで0へ消え、符号精度とpermutationもchance相当だった。

## 次

追加実行は行わない。短距離の弱いsignalはfixed-horizon recoveryを直接監査するexp284の補助解釈に限定する。
