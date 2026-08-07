# exp266 結果

## 結論

`11d0f5ac`でPF ANCC / PF-ZがHMM・likelihood-PFより大幅に良かった現象は、単一seedの偶然ではない。
PF ANCCの元seedは新規63 seed分布のほぼ中央値、PF-Zの元seedはむしろ悪い側95.2 percentileにあり、
新規seedでもHMM・likelihood-PFへのstrong marginを両手法とも63 / 63 seedで再現した。

ただし、元seedで同じstrong phenotypeに入った53 wells全体は均質ではない。両手法で過半数seedが
strong marginを再現したのは21 wells、再現率80%以上は11 wells、全seed再現は4 wellsだった。
絶対RMSE 5 ft以下を両手法で80%以上再現したのは`11d0f5ac`と`fb0904bd`の2 wellsだけである。
したがって、現象は他wellにも存在するが、`11d0f5ac`は特に強く安定した稀な例である。

## 実行契約と完了状態

- 親: exp072 deterministic v2。
- Route: `pf_beam`。
- 対象: 3,783,989 rows / 773 wells。
- PF ANCC / PF-Z: 各600 particles、元seed 1 + 新規stable seed 63、計64 seed。
- PF dynamics variant / LightGBM config / fold / booster: 2 / 0 / 0 / 0。
- Kaggle CPU version 3、8 workers、GPU・親再学習・inference・submissionなし。
- runtime: 12,482.144秒（3時間28分02秒）。parity 218.579秒、multiseed 11,958.578秒。
- seed 0 exact parity: PF ANCC / PF-Zとも3,783,989行、max差0、nonzero 0。

## `11d0f5ac`のseed安定性

比較基準のRMSEはPF ANCC 2.382190、PF-Z 3.386312、exp226 2.874875、HMM 21.161156、
likelihood-PF 24.438100だった。

| 指標 | PF ANCC | PF-Z |
| --- | ---: | ---: |
| 元seed RMSE | 2.382115 | 3.386387 |
| 元seed lower-tail percentile | 0.507937 | 0.952381 |
| 新規63 seed RMSE mean / std | 2.509247 / 0.719886 | 2.522810 / 0.625728 |
| 新規seed RMSE q10 / median / q90 | 2.333788 / 2.379021 / 2.475144 | 1.785696 / 2.725870 / 3.316659 |
| RMSE 5 ft以下 | 62 / 63（98.41%） | 63 / 63（100%） |
| 5 ft成功率 Wilson下限 | 91.54% | 94.25% |
| exp226に勝つseed | 60 / 63（95.24%） | 43 / 63（68.25%） |
| HMM / likelihood-PFに勝つseed | 63 / 63 / 63 / 63 | 63 / 63 / 63 / 63 |
| strong margin再現 | 63 / 63 | 63 / 63 |
| 元seedとの終端誤差符号一致 | 98.41% | 100% |

PF ANCCの元seedは典型的であり、幸運seedではない。PF-Zは新規seedの95.2%が元seed以上に良いため、
元seedが良すぎたのではなく、むしろ元seedが悪い側に外れていた。`>1000 ft`のlong tailでも、64 seed
mean pathのRMSEはPF ANCC 1.5700、PF-Z 1.9905で、終端までの優位が局所区間だけの見かけではない。

## 他wellへの再現

元seedでstrong phenotypeだった53 wellsについて、新規63 seedで同じmarginを再現する割合を監査した。

| 条件 | PF ANCC | PF-Z | 両手法 |
| --- | ---: | ---: | ---: |
| 過半数seedで再現 | 38 | 27 | 21 |
| 再現率80%以上 | 20 | 23 | 11 |
| 再現率90%以上 | 17 | 20 | 9 |
| 全63 seedで再現 | 10 | 13 | 4 |

両手法で全seed再現した4 wellsは`11d0f5ac`、`bb682ebd`、`f0188a48`、`fb0904bd`である。ただし
`bb682ebd`と`f0188a48`はHMM/likelihood-PFがさらに悪いためrelative marginが強いwellであり、絶対誤差が
小さいという意味ではない。両手法で80%以上のseedがRMSE 5 ft以下だったのは`11d0f5ac`と`fb0904bd`
だけ、10 ft以下まで広げると9 wellsだった。絶対精度と他手法に対するrelative marginは分けて扱う必要がある。

一方、PF ANCC strong 53 wellsでは元seedが下位10 percentile以内のwellが10 / 53（18.9%）あり、
nonstrong 720 wellsの84 / 720（11.7%）より多かった。PF ANCCのstrong groupは元seed RMSE平均9.135から
新規seed中央値平均12.075へ悪化し、64 seed meanが元seedを改善したのも22 / 53 wellsだけだった。
元seedで抽出したstrong groupにはselection-on-seedが含まれる。`11d0f5ac`の結論を53 wells全体へ
そのまま一般化してはいけない。

## 発生条件の多角的監査

raw / target-free条件だけでstrong phenotypeを切る明瞭なthresholdは得られなかった。

- eval tail長は新規seed RMSE平均と弱い正相関を持った（PF ANCC ρ=0.219、PF-Z ρ=0.226）。最長20%では
  新規seed中央値RMSE平均がPF ANCC 12.952、PF-Z 16.684となり、seed分散も増えた。
- ただしtail長とstrong margin再現率の相関はPF ANCC -0.062、PF-Z -0.103に留まり、strong 53 wellsの
  eval rows中央値4,657はnonstrongの4,847.5より長くない。長いtailは不安定化条件だが発生条件そのものではない。
- known rows、typewell rows/range、GR sigma、初期rate、PF-Z beta/intercept/sigmaとの関係は弱いか非単調で、
  分布も強く重なった。PF-Z終端符号一致と`pf_z_sigma`だけがρ=-0.198で、単独gateには弱い。
- target/reference-awareには、HMM RMSEとstrong再現率の相関がPF ANCC 0.549 / PF-Z 0.460、
  likelihood-PF RMSEでは0.471 / 0.410だった。発生は「PFが特別に低誤差になるraw regime」というより、
  HMM / likelihood-PFが外れるrelative-method regimeとPF軌道のlong-tail追従が重なる現象として説明しやすい。

最も不安定なwellではPF ANCC seed RMSE stdが18 ftを超えた。強い元seed phenotype内にも
`8f201368`、`368131f9`、`a6f967fb`のようなstd 14 ft超があり、単一seed候補をwell gateなしで信頼する設計は危険である。

## seed集約

全773 wellsのpooled RMSEでは固定meanがmedian / 10% trimmed meanを上回った。

| 手法 | seed 1 | 4 | 8 | 16 | 32 | 64 mean | 64 median | 64 trimmed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PF ANCC | 14.493051 | 13.126896 | 13.027107 | 12.924443 | 12.885644 | 12.830319 | 13.461560 | 12.963931 |
| PF-Z | 17.788171 | 17.178861 | 17.201465 | 17.153356 | 17.124414 | 17.074522 | 17.399634 | 17.178546 |

PF ANCCは4 seed meanだけで64 seed改善量1.662731 ftの約82%を回収した。PF-Zも4 seedで改善の大半を
回収するが、既存exp106/104のPF-Z seedbag系より弱いため、この結果だけでPF-Zを再度portしない。
次に再生成を検討するなら、PF ANCCの固定4/8 seed meanを低コストcandidateとして、既存bankとの
unique oracle・distance/hidden-like・worst-well・current-test実行時間を先に監査する。

## 失敗版と再現性

- v1は数字型well IDのpandas mixed dtypeで776 wellsとなり、PF前にfail-closedした。
- v2はexp072 float32 PF列をCSV decimalのfloat64として比較し、最大0.000484 ftでparity停止した。
- v3はIDをparse時string固定し、PF列を元のfloat32へ復元して全行exact parityを達成した。
- 必須12 artifactsはbytes、raw SHA、decompressed SHAを全件照合した。seed-by-well 98,944行、
  aggregate 27,828行、stability 1,546行、strong path詳細494,204行を確認した。
- input manifest SHA: `7ae1df7457b9ef0bd454d1a0a3620a62a1d363ecb5b45bc255fcf594ddc323d0`。
- artifact manifest SHA: `440a5e474b19290d667562c242ff7aec73420f6340e712d34d01a89bf45cd69c`。
- Kaggle v3実行config SHA: `11e4b8e588c7b431d854695a43bb9f1266a976ee16c136cc080b59f643ee1b0c`。

## 判断

exp266はtrain-side stability auditとして完了する。`11d0f5ac`の偶然性仮説は棄却し、PF ANCC/PF-Zの
構造的な優位を支持する。一方、53-well strong groupにはseed選択バイアスと大きな異質性があるため、
単一seed hard gate、元seed phenotypeによる推論時選別、直接submissionには進めない。
