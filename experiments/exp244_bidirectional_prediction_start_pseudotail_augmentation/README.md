# exp244_bidirectional_prediction_start_pseudotail_augmentation

## 状態

- Route: `ml_model`
- 状態: v4 integrated GPU train v1完了・worst-well guard失敗で不採用
- official-start OOF: raw `8.475793752` / v4 `8.472379731`（`-0.003414021`、総合微改善）
- Public / Private LB: 未提出
- 親実験: `exp239_distribution_matched_multicut_pseudotail`
- model比較親: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 作成日: 2026-07-13

## 仮説

official prediction startより前だけを使うexp239のearly-start pseudo-tailに、official controlと
train-only late-startを加える。異なるprefix長とre-anchor位置を同じfold-safeなmulti-viewとして
学習候補へ供給すれば、learned likelihood、candidate ranker、small calibratorがlong-tailと
re-anchor後回復の両方に頑健になる可能性がある。

## 変更点

- official cutoffから`-1000/-250/0/+250/+1000` rowsの固定start候補を作る。
- startを`early`、`original`、`late`へ分類し、lateはtrain-onlyとする。
- source well単位の5-foldを全派生viewへ継承し、v2ではexp218 official-tail row-weighted foldへ固定する。
- 各startで`TVT_input`をraw true TVTから再構成し、tail true TVTをfeature builderへ渡さない。
- anchor、prefix GR、trajectory、距離特徴を最大1,000 rows/viewでmaterializeする。
- PF/Beam、learned likelihood、GRWR、PF/HMM初期状態の後続再生成contractを保存する。
- current-testではactual start以前のearly viewだけをcalibration backtest requestにする。

## v4 本来の統合学習

v3のlocal-linear confidence shrinkは補助枝であり、本来のmulti-view学習ではなかった。v4では
originalとしてexp239 official 380-feature cacheの全3,783,989行を使い、`-1000/-250/+250/+1000`
のstart位置で380特徴をrawから再生成したpseudo rowを、同じexp218-family LightGBMへ直接加える。

- original: 3,783,989 rows、weight 1.0
- early/late: 3,081 views / 770,157 sampled rows、weight 0.5
- offset別: `-1000` 764 / 191,000、`-250` 773 / 193,250、`+250` 773 / 193,157、
  `+1000` 771 / 192,750（views / rows）
- pseudoは5距離帯から各50行、最大250行/viewを決定的に選ぶ。
- outer-valid source well由来pseudo rowは各foldのtrainから除外する。
- validationはofficial-start 3,783,989行だけ。保存済みexp218 OOFをcontrolとして再利用する。
- 学習量は1 variant / 3 configs / 5 folds / 15 boosters。親/controlは再学習しない。

4つのCPU feature-cache notebookとmemmap streaming GPU train notebookを実装し、raw 773 wellsで
view数・行数を検算した。Jupytext、ruff、py_compile、strict experiment validation、Kaggle package
prepare、repository pytest 15件はpassしている。offset cache v1は4本とも完了し、共通380-feature
schemaと期待3,081 views / 770,157 rowsを確認した。cache SHAをpinしたうえで、承認済みGPU train v1を
`kentookumura/exp244-bidirectional-multiview-train`として完走した。

v4 OOFは8.475793752から8.472379731へ0.003414改善した。1000+は-0.009135、hidden-like spatialは
-0.415836、typewell-purgedは-0.405110で、3 / 5 foldsが改善した。一方、387 wells改善 / 386 wells悪化、
14 wellsが+2 ft超悪化し、最悪`059c8f24`は+16.650567だった。worst-well guard失敗により
`adoption_supported=false`であり、inference / submissionへ進まない。

## 検証方針

- primary: 後続のofficial-start OOF RMSE
- group: source well GroupKFold
- stress: start方向、残tail長、1000+、hidden-like、worst-well
- leakage: late true TVTはtrain-only。outer-valid source well由来viewは対応foldのtrainへ入れない。

## 初期実装範囲

CPU deterministic manifest / prefix feature auditだけを実装する。active audit 1、LightGBM config 0、
fold学習0、booster 0、親/control再学習なし。Kaggle push、推論予測、提出は行わない。

## 所見

Kaggle CPU train audit v1は773 wellsから3,854 views、3,850,880 rowsを生成し、fold、late
train-only、unknown-tail禁止、full-prefix cache slice禁止、materialization coverageをすべてpassした。
early/original/lateのview shareは0.398806 / 0.200571 / 0.400623で、late上限0.45以内だった。
inference audit v1は3 test wellsに`-1000/-250`の6 known-prefix requestsを作り、3,750 rowsを
materializeした。actual start超過、unknown-tail参照、full-model fine-tune、submission生成guardはpassした。
ただしofficial-start OOFは未取得なので、現時点では採用候補でもdeterministic anchorでもない。

parity guard v2はexp218 frozen OOF 3,783,989 rows / 773 wells、RMSE 8.475793978、
OOF/model manifest/raw official surfaceのidentityを確認した。exp244 v1実foldとは174 wellsのみ一致し、
599 wellsをexp218互換foldへ変更した。これは評価面の修正であり、calibrator性能の支持ではない。

v3はknown-prefix dual-start local-linear errorで最大5%のconfidence shrinkを行ったが、overallは
8.475793978から8.477243182へ+0.001449悪化した。1000+、hidden-like 2面、4 / 5 foldsも悪化したため、
current-test portや追加gridへ進まない。

v4は本来の位置変更学習を完走し、overallとhidden-likeでは改善した。しかしfold 1は+0.909638、
fold 3は+0.132699、worst wellは+16.650567と不安定だった。schema・cache・fold leakage・15 model・SHAの
contractはpassしているため、実装不良ではなくmixed early/late augmentationの不均一な効果と判断する。
early/late同時投入のため方向別の寄与は未識別であり、aggregate微改善だけで統合効果を採用しない。

## 実行入口

- train audit: `exp244_bidirectional_prediction_start_pseudotail_augmentation_train.ipynb`
- test calibration request audit: `exp244_bidirectional_prediction_start_pseudotail_augmentation_inference.ipynb`
- canonical train kernel: `kentookumura/exp244-bidirectional-pseudotail-augmentation-train`
- canonical inference kernel: `kentookumura/exp244-bidirectional-pseudotail-inference`
- parity guard: `exp244_bidirectional_prediction_start_pseudotail_augmentation_guard.ipynb`
- canonical parity kernel: `kentookumura/exp244-frozen-anchor-parity-guard` v2
- v4 cache notebooks:
  `exp244_bidirectional_prediction_start_pseudotail_augmentation_multiview_cache_{m1000,m250,p250,p1000}.ipynb`
- v4 integrated train:
  `exp244_bidirectional_prediction_start_pseudotail_augmentation_integrated_train.ipynb`
- prepared cache kernels: `kentookumura/exp244-multiview-cache-{m1000,m250,p250,p1000}`
- prepared GPU kernel: `kentookumura/exp244-bidirectional-multiview-train`

## 採用条件

v4 integrated OOFがraw exp218 OOFよりoverallで改善し、1000+とhidden-like 2面が非悪化、
worst-well回帰+2 ft以内、5 folds中3 folds以上改善すること。late short-tailやpseudo surfaceだけの
改善では採用しない。

## 次

本branchは停止する。再開する場合は同じcache・sampling・weightでearly-only / late-onlyを分離する
matched attributionを先に行い、lateの独立した補償効果とworst-well安全性を確認する。追加学習は
2 variants / 3 configs / 5 folds / 30 boosters相当になるため、実行前に別途明示承認を得る。
