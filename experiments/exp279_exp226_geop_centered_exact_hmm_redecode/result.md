# exp279_exp226_geop_centered_exact_hmm_redecode 結果

## 状態

Kaggle private CPU version 1でfull OOFを完了した。入力、coverage、exp263 parity、有限値、
artifact SHAはPASSしたが、性能guardは全てFAILした。事前契約どおりparameter救済や
inferenceへ進まず、branchをnegative resultとして閉じる。CV / LB anchorは更新しない。

## 仮説

exp226 geometryを毎行の弱いabsolute referenceとしてexact HMM内部へ加えると、GR matchingが
別modeへ逸脱した後も前状態のoffsetを引きずる現象を抑えられる。

## 設定

- 科学的親: exp209 exact HMM
- geometry input: exp226 group-safe OOF `tvt_geop`
- unary: Gaussian `sigma=20 ft`、`lambda=0.50`、clip 600、1 fixed variant
- HMM: exp209 grid / 41 rates / transition / GR emission / calibration / missing-GR固定
- 検証: 5-fold group-safe unknown suffix OOF、3,783,989 rows / 773 wells
- 比較: exp226、exp209 exact HMM、exp263 fixed formula
- 実行量: 1 HMM variant、773 well-runs、0 LightGBM config、0 trained fold、0 booster

## 実行

- kernel: `kentookumura/exp279-exp226-geop-exact-hmm-redecode-train`
- Kaggle id_no / version: `127766774` / `1`
- runtime: 18,663.389秒（約5時間11分）
- private CPU、GPU / TPU / internet off
- inference / submission: disabled / disabled

## Overall結果

| candidate | RMSE | exp263との差 |
| --- | ---: | ---: |
| exp226 prediction | 9.427110 | +1.188778 |
| exp209 exact HMM | 11.938287 | +3.699955 |
| exp263 fixed baseline | 8.238332 | 0.000000 |
| `geop_hmm` | 10.035987 | +1.797655 |

`geop_hmm`はexact HMM単体からは1.902300 ft改善したが、採用対象のexp263 fixed baselineを
大幅に下回った。well単位では267 / 773 wellsで改善、506 / 773 wellsで悪化し、中央値差は
`+0.971711 ft`だった。

## Promotion guard

| check | 結果 |
| --- | --- |
| exp263 RMSE parity、許容1e-5 ft | PASS（差 `7.45e-7 ft`） |
| overall gain 0.02 ft以上 | FAIL（gain `-1.797655 ft`） |
| 改善fold 3 / 5以上 | FAIL（0 / 5） |
| near悪化0.02 ft以下 | FAIL（`+0.198846 ft`） |
| 1000+悪化0.02 ft以下 | FAIL（`+1.983953 ft`） |
| hidden-like spatial悪化0.02 ft以下 | FAIL（`+1.381884 ft`） |
| hidden-like typewell-purged悪化0.02 ft以下 | FAIL（`+1.451614 ft`） |
| worst-well悪化+0.25 ft以下 | FAIL（`+27.158481 ft`、well `389ae58f`） |
| geometry grid / finite coverage 100% | PASS / PASS |

Fold別の悪化は`+2.848074 / +1.593594 / +2.103474 / +1.272560 / +1.248048 ft`で、
方向の不安定さではなく全fold一貫した回帰だった。

## Persistent-offset recovery

誤差10 ft超が128行以上連続したepisodeを、確認後256 / 512行以内に誤差5 ft以下へ戻るかで評価した。

| candidate | episodes | 256行以内 | 512行以内 |
| --- | ---: | ---: | ---: |
| exp226 prediction | 645 | 1.71% | 5.89% |
| exp209 exact HMM | 638 | 2.04% | 10.19% |
| exp263 fixed | 551 | 2.18% | 9.07% |
| `geop_hmm` | 802 | 2.62% | 11.85% |

`geop_hmm`は復帰率だけならわずかに上がったが、episode数がexp263比で251件増えた。
したがって「absolute referenceが一度外れたpathを戻す」効果は部分的には見えるものの、
新たな持続誤差を増やさず全体を改善する安全な補正にはなっていない。

## 再現性 / technical audit

- 773 / 773 well status `ok`、3,783,989 / 3,783,989行を生成・評価。
- exp226 / exp209 / exp072 decompressed input SHAは事前契約と一致。
- summary記載の11 artifact SHAは取得ファイルと全件一致。
- OOF raw gzip SHA: `2f94f4977004d35c3ce443dffe241d3500a250042ece120b05114dc795732d6b`
- OOF decompressed SHA: `9e29e70783eb13abff5ccbee23acbb274589457f233fdb266094a4db0a24d7c0`
- logical prediction SHA: `335ba03183d5c18140d237a2d9adf7a39bb5fdb5dd00c56c507dcd59bc9e613c`
- decoder scientific manifest SHA: `0a3af93a14376925e08340265fb42a6f694203ac5fc131087bb4675f22ecc021`
- summary file SHA: `b4332b3c72da37447bfc45cf4b974a45b4571682a46346da671374fed6e9ae87`
- deterministic anchor: いいえ。成功runはversion 1の1回だけ。

## 解釈

HMMはGR emissionとtransitionの累積posteriorを前行から伝えるため、似たGR modeへ移ると局所形状を
保ったままoffsetを持続できる。今回のgeometry unaryはその履歴を弱め、exact HMM単体より改善した。
一方、exp226 geometryはGRとは独立でもwellごとの絶対位置誤差を持ち、固定強度のunaryは正しい
GR evidenceまで誤ったgeometryへ引く。これが全foldと長距離・hidden-likeでの一貫した回帰、
episode増加、worst-well悪化として現れた。単に「前stepを引きずる」だけでなく、戻り先の
absolute anchor自体の不確実性と固定weightが主な限界である。

## 次

事前契約に従いsigma / lambda / grid / process-noise探索、exp226最終予測unary、PF併用、blend、
本branch内のselector、raw-test inference、submissionは行わず、直接枝を閉じる。
ただし2026-07-19のユーザー指示により、exp279の直接救済とは分離した後続仮説として、
修正版exp264へ`geop_hmm`を疎な13番目のadd-only candidateとして接続するbacklogを
`backlog/KAGGLE_DIRECTION.md`へ追加した。既存exact/self-GR HMMの置換や固定blendは行わない。
