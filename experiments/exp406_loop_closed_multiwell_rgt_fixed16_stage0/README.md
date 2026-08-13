# exp406_loop_closed_multiwell_rgt_fixed16_stage0

## 状態

- Route: `pf_beam`
- 状態: Kaggle CPU fixed16 Stage 0 version 1 technical FAIL・branch閉鎖
- 開始条件: exp405の技術的に有効なscientific FAIL
- Stage: fixed16 Stage 0のみ
- CV / LB: なし
- 設計の正:
  `docs/legacy/steering/20260726-exp406-loop-closed-multiwell-rgt-fixed16-stage0/`

## 仮説

horizontal GRのpairwise局所対応を全well graphで同時にloop-closeすれば、
exp386のFormation-derived graphで0だったtarget query/path supportを、
current-testでも観測可能な入力だけで非退化にできる。

## 固定方式

- target: exp386と同じround-robin fold selectorによる16 wells
- block / stride: H256 / H128
- donor: outer-train、same-Type-Well優先、最大12 wells
- local search: exp226 geometry中心`±55 ft / 5 ft`
- GR: raw / rolling-21 / rolling-101
- edge: blockごと上位4
- loop closure: TVT ft単位、fundamental cycles、Huber IRLS 10回
- readout: graph support、cycle residual、circular control、
  visible-prefix末尾512 rows rolling-origin、resource
- unknown suffix prediction / scenario bank: なし

## exp386との差

Formation 6列からRGT nodeを作らず、target GRを観測としてpairwise edgeを先に作る。
k-shortest routeや8--32 scenariosは列挙しない。exp386から再利用するのは
fixed16 selectorだけで、threshold緩和による救済ではない。

## 検証方針

graph query、connectedness、finite gauge、cycle residual、real対circular、
prefix512 rolling-origin、full runtime/RSS投影を全ANDで判定する。
unknown suffix truthは読まず、suffix pathも保存しない。

prefix512のexp226 controlは、保存OOFがofficial suffix行だけでprefixを覆わないため、
ユーザー承認済みのfixed16限定replayを使う。target-free graph freeze後に
outer-trainからoriginal K16 geometry field / adaptive Kappaをfold別再構築し、
pseudo-cutの`tvt_geop`相当だけを生成する。target ANCC、GR correction、
U-projection、official OOF再生成は行わない。

## 合格後

Stage 0全gate PASS時だけ、同じexp406内のfull-OOF Stage 1を別設計・別承認で
追加できる。Stage 0 PASSはcurrent-test実装、inference、submissionの承認ではない。

## 所見

exp386の失敗はRGT source coverage不足ではなく、graph query 0とcycle residualの
単位・経路問題だった。exp406はFormation graphを修正せず、観測sourceをGRへ
置換した独立仮説として扱う。

## 実装状態

- `*_compact_selfcontained_train.py` / `.ipynb`: 10章の実装候補
- `*_compact_selfcontained_inference.py` / `.ipynb`: fail-closed候補
- 専用test: 13件PASS
- 正規train Notebook: compact self-contained候補を採用
- 正規inference Notebook: placeholderのまま未採用
- Kaggle package / push / run: version 1完了
- full OOF / current-test / inference / submission: 無効

## 実行結果

- decision: `close_exp406_without_parameter_rescue`
- technical: 12/15 PASS
- graph query coverage: `0.451157 < 0.90`
- finite loop-closed row coverage: `0.755026 < 0.95`
- projected runtime: `65,543.109 > 30,600 sec`
- real-circular NCC: `+0.874148`、5/5 foldsでreal優位
- target-free gateで停止したためprefix replayは未実行
- unknown suffix prediction、model、PF/HMM/Beam、submissionは0

## 次

exp406はparameter rescueなしで閉じる。full OOF、current-test、inference、
submissionへ進まない。exp386 route棄却の独立原因分解はP4候補として別管理する。

## 表記

用語は`backlog/KAGGLE_DIRECTION.md`と`docs/glossary.md`に合わせ、
実験名・設定名以外は日本語優先で記録する。
