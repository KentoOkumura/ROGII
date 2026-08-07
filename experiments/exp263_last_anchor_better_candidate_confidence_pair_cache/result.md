# exp263_last_anchor_better_candidate_confidence_pair_cache 結果

## 仮説

anchorより良い候補を33本のwide bankとして後続ごとに再収集するのではなく、family圧縮した12
primitive、実在するtarget-free confidence、8 pair、3 named combinationへ固定すれば、候補bankの
定義揺れ、target由来confidence混入、deployability tier混同、pair tensorの重複を防げる。

## 実行

- Kaggle kernel: `kentookumura/exp263-last-anchor-pair-cache-train` version 1 / id_no `127474050`
- 状態: `KernelWorkerStatus.COMPLETE`
- runtime: private CPU、GPU/TPU/internet off
- cache生成: 951.444秒、metrics保存: kernel開始から981.749秒
- active variant / LightGBM config / fold training / booster: 0 / 0 / 0 / 0
- parent/control再学習、PF/Beam再生成、submission: なし

## Stage 0結果

| 項目 | 結果 |
| --- | ---: |
| 行 / wells / folds | 3,783,989 / 773 / 5 |
| reference / core / raw-test core | 33 / 12 / 6 |
| shortlist pair / raw-test pair / named formula | 8 / 5 / 3 |
| value Parquet | 60 partitions / 2,665,368,732 bytes |
| confidence Parquet | 60 partitions / 515,849,184 bytes |
| source | 9 groups / 12 gzip files、全core候補で全行coverage |
| temporary work files | 削除済み |

主要SHA:

- cache manifest: `85e60ac10b50197fa44ea29faffcbba81bd0746114bc53bae0f5cc537a26bb9e`
- candidate catalog: `7cd748661b719bfcfb1ed21b9fe314366b1c089cc0dd224a9e4cbd4ba7e9e6e0`
- canonical ID: `de07df322a8dc3a981b556ea78c3ddcecb2e708a31e986cfe05873109d4ae9d3`
- generation config: `3cf69b76296441c22a03a67f56e86b0bcfceaf03d2cbdab873deb575e8c0451b`
- pair readout: `402a459e3c939c15aeee04ba8ad8ead957847f55f7ad85c97093987c2a514c11`
- small parity sample: `68d00cba720b55222be64106b6deb456cb997cfe20ccecee38935ee4e3bcf589`

exp072 decompressed SHA `99a3c70a...e1350`、exp209 `ee3b548b...ee3f4`を含め、全12 sourceのraw /
decompressed SHAをmanifestと`metrics.json`へ記録した。

## Pair readout

Stage 0がfull sourceから再計算した8 pairのoverall結果:

| pair | tier | RMSE | better parent比 | 改善 / 悪化 wells |
| --- | --- | ---: | ---: | ---: |
| exp226 + self-GR HMM a070 | raw-test | 8.532715 | -0.894395 | 481 / 292 |
| exp226 + exact HMM | raw-test | 8.635074 | -0.792035 | 473 / 300 |
| exp226 + likPF | raw-test | 8.813822 | -0.613288 | 458 / 315 |
| self-GR HMM a070 + likPF | raw-test | 10.123457 | -1.226486 | 400 / 373 |
| likPF + exact HMM | raw-test | 10.269697 | -1.325201 | 539 / 234 |
| exp226 + peer-atlas HMM | train-only | 8.607484 | -0.819625 | 471 / 302 |
| exp226 + exp192 likPF | train-only | 8.727406 | -0.699704 | 468 / 305 |
| exp226 + K8 medoid m0 | train-only | 8.989907 | -0.437203 | 473 / 300 |

固定`exp226_w500_50_50`はexp226 50% + likPF 25% + exact HMM 25%、契約OOF RMSE 8.238331、
exp226に5/5 foldsで勝つ式としてnamed manifestへ固定し、small parity sampleで実体を生成した。

outer-train-only eligibilityは100 candidate×fold中99件がeligibleだった。`beam_mean`のfold 1だけは
outer-train RMSE 15.911830がanchor 15.898000を上回るためineligibleである。この表は監査専用で、
row featureやraw-test artifactへ混ぜない。

## 生成物検証

- Kaggle outputには143 filesがあり、同名`part-000.parquet` 120件を確認した。
- manifest/catalog/readout/eligibility/named formula/parity/metricsを限定取得した。
- best pair両親のfold 0 value/confidence 4 Parquetを代表取得した。
- 4ファイルすべてでrows、bytes、file SHA、logical content SHA、schema SHAがmanifestと一致した。
- virtual loaderでexp226 + self-GR HMM a070を757,738行再構成し、直接50/50平均との差は最大0だった。
- Kaggle pandas 2系の`object`とローカルpandas 3系の`str`差をhash前に`object`へ正規化するよう
  loaderを補強した。既存manifest値は変わらず、両環境から同じlogical SHAを再検証できる。

## 解釈

Stage 0 cacheは後続OOF実験のcanonical inputとして採用できる。HMM+LGB、selector/TVT outputs、
target/error/oracle confidence、exp104 pair sweep、K8 m1-m6複製、pair/triple full tensorは入っていない。
confidence未提供値も補完せずmissing contractを維持している。

Stage 1 inference v2では6 current-test primitive、5 pair、固定`exp226_w500_50_50`を14,151行で
再構成し、値parityを通した。runtimeは225.459秒、formula parityは最大0、exp237 referenceとの差は
6 primitiveすべて最大0.000484375以内だった。

## Stage 1 hidden-safe inference / submission

| 項目 | 結果 |
| --- | --- |
| inference kernel | `kentookumura/exp263-last-anchor-pair-cache-inference` v2 |
| selected candidate | `exp226_w500_50_50` |
| fixed formula | `0.50*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm` |
| OOF RMSE | 8.238331 |
| rows / wells | 14,151 / 3 |
| runtime | 225.459秒、CPU、GPU/internet off |
| formula / reference parity | 最大0 / 最大0.000484375（許容0.001） |
| prediction SHA | `a418876d319301702cc6c3e28b0d30e95518510ef9c83823197c4ecff2e3ce4b` |
| submission SHA | `6316695197ee67c9a2aaa23754e6f2a5cf30dd0ec4ef1a018921f9ea640a1dbc` |
| submit-check | PASS |
| submission ref / status | `54761954` / `COMPLETE` |
| Public / Private LB | **7.800** / - |

Public LBはOOF RMSEより-0.438331、exp226単体のPublic LB 9.837より-2.037改善した。固定blendの
補完性とhidden-safe再生成は支持された。一方、exp257のPublic LB 7.718より+0.082、全体ensemble
anchor exp082の7.601より+0.199悪いため、全体anchorは更新しない。exp218の7.843よりは-0.043良い。
LBを見たweight gridや係数再調整は行わない。

exp264 Stage Aで21 confidence依存特徴が採用されたため、同じStage 1に21 namespaced confidence列を
追加した。exp226 GR delta、HMM std/loglik/self-GR診断、PF std、Beam family stdは各候補値と同一generator
callから得る。likPFはnative scalarなしを明示し、formulaは親confidenceを平均しない。

## Stage 1 namespaced confidence v3

| 項目 | 結果 |
| --- | --- |
| inference kernel | `kentookumura/exp263-last-anchor-pair-cache-inference` v3 / `COMPLETE` |
| rows / wells / runtime | 14,151 / 3 / 354.341秒 |
| artifact shape | 14,151 rows × 36 columns（v2 15列 + confidence 21列） |
| confidence completeness | 21/21列で全行non-null・finite |
| valid rate | exp226/self-GR HMM/exact HMM/PF-ANCC/Beam 1.0、likPF 0.0（native scalarなし） |
| v2 value parity | 旧15列 exact equality、最大絶対差0 |
| formula / reference parity | 最大0 / 最大0.000484375（許容0.001） |
| extended Parquet SHA | `bda0502894d6a20cc3c332d729cf120b17ceed2e1773093bd7140c6df71e360c` |
| prediction / submission parity | v2 SHAと一致、CSV byte-identical |
| submit-check / competition submit | PASS / なし |

これによりexp264が要求するcurrent-test namespaced confidence契約は満たされた。v3はv2の候補値や
提出予測を変更していないため再提出しない。

## 次

exp263はStage 1まで完了とする。新しいpair探索、weight grid、HMM+LGB再導入、selector学習は
この実験では行わない。selector学習はexp264 Stage Bの別承認で扱う。
