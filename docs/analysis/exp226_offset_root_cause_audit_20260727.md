# exp226 オフセット根本原因監査 2026-07-27

## 結論

exp226 のオフセットは、単一の定数 bias、誤った境界 TVT、行順、CV fold、
K=16 の境界ジャンプ、GR 補正単独、U projection 単独、または公開コードの
移植ミスではない。

主因は次の連鎖である。

1. exp226 は最後の既知 `TVT_input` を一度だけ絶対 anchor とする。
2. unknown suffix では、空間 donor から補間した K16 区間の相対的な
   `TVT + Z` 増分を累積して TVT path を作る。
3. target well の局所構造と donor field の増分が少しずれると、行単位では
   `0.02--0.04 ft` 程度の小さな signed rate errorでも、再 anchor がないため
   長い suffix で積分される。
4. その累積誤差は次の K16 区間へほぼ定数の vertical offset として持ち越される。
5. GR 補正と U projection は pooled ではこの誤差を改善するが、絶対位置を
   一意に復元するほど強くなく、一部 well では proximal trigger として
   10 ft threshold を越えさせる。

したがって、見えている「平行にずれた線」は原因そのものではなく、
局所的・空間的な増分誤差を一度しか anchor しない累積 path に入れた結果である。
donor が遠い well、長い suffix / 大きい TVT range、遠距離 donor bin の不安定な
係数では、この機構が強くなる。

## 対象と符号

- 対象: exp226 group-safe 5-fold OOF
- rows / wells: `3,783,989 / 773`
- OOF decompressed SHA256:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- error 符号: `prediction - true`
- persistent offset: `|error| >= 10 ft` が 128 行以上連続
- K16 区間は exp226 本体と同じ `np.linspace + searchsorted(side="left")`

監査は保存済み OOF を well 単位で stream 集計した。新規学習、exp226 再生成、
inference、submission は行っていない。

## 1. 一律の global offset ではない

| 診断 | 値 |
| --- | ---: |
| final OOF RMSE | 9.427110 |
| global bias | -0.299619 ft |
| global bias を除いた RMSE | 9.422347 |
| global bias が説明する MSE | 0.1010% |
| well ごとの mean offset を除いた RMSE | 5.777591 |
| well mean が説明する MSE | 62.4391% |
| well ごとの affine offset+slope を除いた RMSE | 4.042230 |
| well affine が説明する MSE | 81.6141% |

全体平均はほぼゼロで、fold bias も
`-0.492 / -1.782 / +0.701 / -0.355 / +0.446 ft` と符号が揃わない。
一つの calibration constant、単位変換、全行の上下反転、特定 fold の不具合では
説明できない。

## 2. 誤差は境界付近では小さく、距離とともに成長する

| suffix 範囲 | final RMSE |
| --- | ---: |
| 0--50 | 1.741257 |
| 50--100 | 1.671925 |
| 100--250 | 2.190612 |
| 250--500 | 3.547468 |
| 500--1000 | 5.472891 |
| 1000--2000 | 7.452736 |
| 2000+ | 11.151214 |

well 長を正規化しても、最初の decile は `3.056299`、最後の decile は
`13.292486` だった。最後の既知 TVT anchor が一律に間違っているなら、
最初から同程度の offset が出るはずで、この成長形とは合わない。

final path の隣接行増分誤差 RMSE は `0.035687 ft/row` である。全体平均増分 bias は
`-0.0000416 ft/row` とほぼゼロだが、well / segment 内では符号が持続するため、
独立な white noise ではなく cumulative drift になる。

## 3. K16 区間内では形状がほぼ合い、区間間へ offset が持ち越される

| oracle 診断 | offset/slope 除去後 RMSE | 説明 MSE |
| --- | ---: | ---: |
| K16 segment mean offset | 1.130603 | 98.5617% |
| K16 segment start re-anchor | 2.144846 | 94.8235% |
| K16 segment affine offset+slope | 0.403654 | 99.8167% |
| H256 block mean offset | 0.908421 | 99.0714% |
| H512 block mean offset | 1.558267 | 97.2677% |
| H512 block affine | 0.609647 | 99.5818% |

oracle 係数は truth を使う診断値であり、deployable correction ではない。ただし、
誤差分散が「row ごとの乱雑な外れ」ではなく「区間で共有される低周波 offset/slope」
に集中していることは直接示す。

さらに K16 segment の error は次を満たした。

- segment mean と segment start error の Pearson: `0.981710`
- segment mean と前 segment end error の Pearson: `0.982951`
- segment mean と前 segment mean の Pearson: `0.950857`
- segment 境界での error jump 中央値: `0.008190 ft`
- segment 境界での error jump p95: `0.031782 ft`
- segment start/end が同符号: `84.1850%`

境界で突然ずれるのではない。前区間の終端誤差を連続的に受け継ぎ、次区間内の
小さな slope error がさらに加算される。

## 4. persistent episode は少数行に MSE を集中させる

| 診断 | 値 |
| --- | ---: |
| episodes | 645 |
| episode を持つ wells | 449 / 773 |
| episode rows | 718,744 |
| 全 OOF に占める rows | 18.9943% |
| 全 MSE に占める episode SSE | 82.0073% |
| onset 1行 jump の絶対値中央値 | 0.021148 ft |
| onset 前64行の error rate 絶対値中央値 | 0.017944 ft/row |
| onset から最寄りK16境界までの中央値 | 81 rows |

10 ft を一行で飛び越える不連続や K16 境界 jump が主体ではない。
約 `0.02 ft/row` の緩い drift が threshold に達した後、その符号を長く保つ。
この 19% の行が SSE の 82%を作るため、見た目でも RMSE でも「大きな平行 offset」
として支配的になる。

## 5. geometry、GR、U projection の役割

| stage | RMSE | increment RMSE |
| --- | ---: | ---: |
| geometry `tvt_geop` | 10.077950 | 0.036017 |
| geometry + GR、pre-U | 9.500816 | 0.036432 |
| final post-U | 9.427110 | 0.035687 |

### GR correction

- pooled RMSE: `10.077950 -> 9.500816`
- 5/5 folds、515/773 wells を改善
- row absolute error: 61.64% 改善、38.34% 悪化
- adjustment absolute mean: `2.127778 ft`
- `|delta| >= 3.99` の cap 張り付き: 24.26%
- 必要 correction との Pearson: `0.343638`

GR は有効だが、500-row window / 125-row stride、最大 `±4 ft` の弱い低周波補正である。
公開済み exp280 でも truth-nearest shift の top1 は 18.95%に留まり、exp281 の
always-on residual-offset HMM は exp226 より RMSE を悪化させた。したがって GR には
offset 情報があるが、absolute branch を常時一意に決めるほど強くない。

episode onset では 23.57%が「geometry は10 ft未満だがGR後に10 ft以上」となった。
これは GR が一部 episode の proximal trigger になり得ることを示す。一方、pooled では
明確に改善し、GR単独を根本原因とはできない。

### U projection

- pooled RMSE: `9.500816 -> 9.427110`
- 5/5 folds、482/773 wells を改善
- row absolute error: 50.36% 改善、49.64% 悪化
- adjustment absolute mean / p95: `0.934081 / 2.907332 ft`
- 必要 correction との Pearson: `0.126317`

U projection は robust degree-4 polynomial で `TVT+Z` を平滑化するため、局所 noise は
減らすが absolute datum を観測しない。episode onset の21.86%では最後の threshold
crossingを起こすが、pooled SSE は改善する。これも増幅・近接要因であって主因ではない。

### episode onset の分類

- geometry がすでに10 ft超: 54.57%
- GR 後に初めて10 ft超: 23.57%
- U projection 後に初めて10 ft超: 21.86%

この分類は threshold crossing の順序依存 attribution であり、SSE の厳密な因果分解ではない。
重要なのは、全 stage で K16 mean-offset quotient が MSE の98%前後を説明し、
低周波累積機構が共通している点である。

## 6. donor extrapolation が主要な risk amplifier

well 単位 final RMSE との Spearman は次だった。

| driver | Spearman |
| --- | ---: |
| suffix TVT range | +0.387342 |
| donor distance max | +0.337118 |
| donor distance min | +0.317283 |
| U adjustment absolute mean | +0.234679 |
| GR cap fraction | +0.187585 |
| GR delta absolute median | +0.177525 |

donor distance の quartile 比較:

| driver | bottom quartile RMSE中央値 | top quartile RMSE中央値 | episode well率 bottom / top |
| --- | ---: | ---: | ---: |
| donor min | 4.583804 | 7.969026 | 44.56% / 72.68% |
| donor max | 4.099483 | 7.774613 | 43.52% / 72.68% |

既存の HMM/PF/exp226 well readoutでも、exp226 が30 ft以上でGR系が10 ft以下の
`389ae58f / 70925e23 / ae8959c3` は donor 距離が全体の上位 tail にあり、
誤差は whole-well bias だった。

これは「遠い donor だけが全 offset の唯一原因」という意味ではない。相関は中程度で、
近い donor でも局所構造差は起こる。ただし、コード上の spatial local-linear field、
距離別 kappa、距離 quartile、extreme well の方向が一致するため、geometry 増分誤差の
主要な増幅条件と判断できる。

## 7. kappa と segment 数

近距離で中心となる係数は fold 間で比較的安定した。

- `raw_bin_1`: std `0.024894`
- `raw_bin_2`: std `0.019255`
- `near_strike_committee`: std `0.019361`

一方、遠距離・弱支持の係数は sign flip または大きな range を持つ。

- `raw_bin_3`: range `0.370453`、1 foldで負
- `raw_bin_4`: range `0.388127`、1 foldで負
- `smooth_bin_4`: range `0.755315`、1 foldで負
- `smooth_bin_0`: range `0.230190`、1 foldだけ正

design columns は相関するため個々の kappa を単独で因果解釈できないが、遠距離 regime の
不安定さは donor extrapolation risk と整合する。

exp302 で K16 だけを K12 / K24 に変えた direct RMSE は
`9.551938 / 9.413244` だった。K24 の改善は `0.013865 ft`、3/5 foldsに留まる。
よって K16 の解像度自体は根本原因ではない。segment 境界は累積誤差を観察しやすい単位であり、
誤差を発生させる discontinuity ではない。

## 8. 公開ソース移植ミスを数値で否定

公開保存ソース SHA256:
`fbaa63c1d38295c4fddf80e8ee8d25865005e5cd0ecc6641cc36ad4bad3c000e`

exp226 port SHA256:
`8293e915672d34b67e10b7951b22dfee50c7551c4ed6d98ac988b4ef9d4db4b8`

固定 synthetic seed `20260727` で、公開関数と port 関数を同じ入力へ通した。
次の全てで最大絶対差は `0.0` だった。

- `segment_geometry`
- `fit_coeffs`、rho 0 / 10
- `local_linear`
- `kernel_mean`
- `build_columns`
- `affine_cal`
- `project_u`
- `gr_correction`（内部 `emissions/decode` を含む）

定数、anchor、K16、distance bin、kappa regimes、GR window/stride/cap、
U projection設定も一致した。OOF row/error parity、重複なし、5 folds、
artifact SHAも確認済みである。

これは deterministic v6 core の移植ミス説を強く否定する。ただし公開 notebook の
v7 neural committee と v8 LightGBM meta-layer は外部 weight がないため exp226 の
契約上無効である。公開完全版には、まさに geometry / GR / PF disagreement から残差を
補正する learned layer がある。したがって「公開完全版にある learned offset correctionを
再現していない」ことは exp226 v6 の性能上限の一部だが、v6 port のバグではない。

## 9. 既存の独立実験との整合

### exp228: row-wise residual LightGBM

exp226 `9.427110 -> 8.944086` と改善した。offset に target-free featureで予測できる部分が
あることは支持するが、exp218 ML anchorには届かなかった。

### exp285: prefix-masked offset predictability

known prefix 末尾640行から official suffix offset を予測する pooled Spearman は
`-0.004135`、sign balanced accuracy は `0.488567`。境界前の offset をそのまま
well 全体へ外挿する案は成立しない。これは「正しい anchor がない」のではなく、
未知 suffix へ入ってから donor 増分差が累積するという結論と合う。

### exp280 / exp281: GR shift と residual-offset HMM

exp280 は raw-GR/typewell shift bank の rank signalを確認したが top1は18.95%だった。
exp281 はMAEやpersistent recoveryを改善した一方、RMSEは `9.827420`でexp226より悪く、
一部wellに誤offsetを維持した。GR evidenceは補助情報であり、absolute anchorの代替ではない。

### exp298: local-shape quotient

H256/H512 affine quotient は絶対値として小さかったが、fixed candidate bank内では
rank 4/5で、pre-U pathを新しい局所shape sourceへ昇格する条件はFAILした。
「offsetを除けば完全に最良」ではなく、exp226の誤差が低周波に集中することと、
他候補より局所shapeが優れることは別問題である。

### exp333: K16 segment residual target

独立実装のK16 oracle mean-offsetも RMSE `1.130603`で本監査と完全一致した。
strict nested segment modelは exp226を5/5 foldsで `0.350433 ft`改善したが、
oracle headroomの大半は回収できず、worst wellを `+8.099 ft`悪化させた。
区間offsetは実在するが、current-testで安全に値を知る問題は別に難しい。

## 10. 否定または単独原因として不十分な仮説

| 仮説 | 判定根拠 |
| --- | --- |
| global calibration bias | MSE説明0.10%、fold biasの符号不一致 |
| last-known TVT anchorの誤り | 0--100行RMSE約1.7、距離とともに成長 |
| row order / submission mapping | OOF key/row連続性、重複、error parity、submission check PASS |
| 特定CV fold | 5 foldsすべて同程度に発生 |
| K16境界jump | 境界jump中央値0.008 ft、onset境界距離中央値81行 |
| K=16という解像度 | K24改善0.014 ftのみ、K12悪化 |
| GR単独 | pooledで0.577 ft改善、ただし一部trigger |
| U projection単独 | pooledで0.074 ft改善、ただし一部trigger |
| source-port bug | 9数値核で最大差0 |
| known prefix offsetの長期外挿 | exp285 Spearmanほぼ0 |
| always-on GR offset decoder | exp281でRMSE悪化とtail発生 |
| unknown truth leakage | target wellはdonor field/kappaから除外、truthはreadoutのみ |

## 因果階層

### 根本機構

`one absolute boundary anchor + spatially transferred relative increments +
cumulative integration + no later absolute re-anchor`

### 誤差を発生させる直接要因

- donor fieldとtarget local structureのsigned rate mismatch
- global kappaがwell固有の局所勾配を完全には表現できないこと
- `TVT+Z` pathをK16 piecewise slopeで近似する残差

### 増幅条件

- donor距離が大きい
- suffix / TVT rangeが長い
- 遠距離binの弱支持・fold不安定
- GR absolute matchingが曖昧またはcapへ張り付く
- U projectionが誤った低周波shapeを滑らかに維持するwell

### 防止しきれない理由

- GR correctionは sparse、capped、shift識別が弱い
- U projectionはsmoothness priorであってabsolute observationではない
- v7/v8 learned residual weightsをexp226では利用していない
- prefixからfuture offsetの符号・大きさを安全に予測できない

## 証拠の限界

- oracle mean/affine除去は誤差構造を示すだけで、test-time correctionではない。
- donor距離との関係はコード機構・quartile・extreme wellが一致するが、単一変数の
  randomized causal interventionではない。
- synthetic parityは deterministic v6 数値核を確認する。公開外部weightを含む
  v7/v8完全実行との同値性は主張しない。
- episode onset attributionは処理順に依存し、geometry/GR/UのShapley分解ではない。

以上を踏まえても、根本機構についてはコード上の累積式、境界からの誤差成長、
K16/H512 quotient、segment間継承、episode onset形状、donor距離依存、独立実験の
全てが同じ方向を示している。

## 生成物

- 実行コード:
  `studies/exp226_offset_root_cause_audit.py`
- 生成物ディレクトリ:
  `studies/exp226_offset_root_cause_audit_20260727/`
- 機械可読要約:
  `summary.json`
- stage / component:
  `stage_metrics.csv`, `component_effects.csv`
- offset quotient:
  `oracle_quotient_metrics.csv`, `k16_segment_statistics.csv`
- persistent episode:
  `persistent_offset_episodes.csv`, `persistent_offset_summary.csv`
- well driver:
  `well_root_cause_readout.csv`, `well_driver_correlations.csv`,
  `driver_quantile_contrasts.csv`
- reproducibility:
  `kappa_fold_stability.csv`, `source_port_contract.csv`,
  `source_port_numeric_parity.csv`

再実行:

```bash
.venv/bin/python studies/exp226_offset_root_cause_audit.py
```

メモリ制約下でも同じ結果を再生成できるよう、OOFはwell連続性を検証しながら
100,000行 chunkで読む。
