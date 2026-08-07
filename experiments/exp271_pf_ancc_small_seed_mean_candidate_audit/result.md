# exp271 PF ANCC small-seed mean candidate audit 結果

## 仮説

exp266固定seed順の先頭4 seed meanが、8 seed meanと同程度のexp263 core bank追加headroomを持つなら、
低コストPF ANCC candidateとして残せる。

## 設定

- 親: `exp266_pf_ancc_pf_z_multiseed_stability_audit`
- candidate bank: exp263 Stage 0 core 12
- PF: exact exp266 PF ANCC、600 particles、固定先頭8 seed、mean4 / mean8
- 対象: 3,783,989 rows / 773 wells
- 監査: row、block 128/256/512、whole-well、distance bucket、hidden-like、worst-well
- 実行量: 1 PF dynamics / 0 LightGBM config / 0 fold / 0 booster
- Kaggle: private CPU、GPU/internet off、kernel version 2

## 変更点

exp266のPF ANCC kernel・particles・seed namespaceは固定し、先頭4/8 seed mean pathの保存と
exp263 core-bank追加headroom監査だけを新設した。PF-Z、selector、inference、submissionは加えていない。
target-free candidate gzipを書いた後にraw horizontal `TVT`をfloat64で読み、評価だけへ結合した。

## 実行結果

version 1はPF生成完了後、exp072のfloat32 `target + last_known_tvt`復元値と、exp266が使うraw
`TVT`の精度差によりper-well parityが最大0.000459 ftずれてfail-closedで停止した。許容値は緩めず、
raw `TVT`評価へ修正したversion 2を同じcanonical kernelへpushし、完了した。

| 項目 | 値 |
| --- | ---: |
| Kaggle status | `COMPLETE` |
| total runtime | 1,386.570 sec（23.11分） |
| PF generation | 953.056 sec（15.88分） |
| seed0 max abs diff vs exp072 | 0.0 ft |
| mean4 max per-well RMSE diff vs exp266 | 7.105e-15 ft |
| mean8 max per-well RMSE diff vs exp266 | 7.105e-15 ft |
| input manifest | 1,611 records、exp263 partition 60、raw horizontal/typewell各773 |
| inference / submission | disabled / 未提出 |

## Standalone

| candidate | RMSE | MAE | bias | within10 |
| --- | ---: | ---: | ---: | ---: |
| exp072 PF ANCC seed0 | 14.493051 | 8.921559 | -1.167419 | 0.691732 |
| PF ANCC mean4 | 13.126896 | 8.376810 | -0.918139 | 0.707970 |
| PF ANCC mean8 | 13.027107 | 8.341032 | -1.006339 | 0.708564 |

mean8はmean4よりRMSEを0.099789 ft改善した。mean4でもseed0から1.366155 ft改善し、
8 seedに増やした追加効果は小さい。

## Exp263 core 12への追加headroom

| oracle scope | core12 | +mean4 | delta | +mean8 | delta | +both | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| row | 2.986502 | 2.939958 | -0.046543 | 2.936781 | -0.049720 | 2.921250 | -0.065252 |
| block 128 | 3.044897 | 2.999673 | -0.045224 | 2.996804 | -0.048093 | 2.981693 | -0.063204 |
| block 256 | 3.119791 | 3.076605 | -0.043186 | 3.072489 | -0.047303 | 3.058377 | -0.061414 |
| block 512 | 3.264960 | 3.223488 | -0.041472 | 3.219914 | -0.045046 | 3.206153 | -0.058807 |
| whole-well | 4.609190 | 4.580799 | -0.028392 | 4.572218 | -0.036973 | 4.558439 | -0.050751 |

mean4単独はrow unique-best 252,772 rows（6.6800%）、mean8単独は251,635 rows（6.6500%）で、
候補追加価値はほぼ同じだった。両方を入れるとmean4 165,283 + mean8 175,404 = 340,687 rows
（9.0034%）がcore12を一意に上回り、whole-wellでも59 wellsを改善した。oracleなのでcandidate追加の
上限を示すだけで、deployable selector性能ではない。

## Stress readout

- hidden-like spatial RMSEはmean4 14.465755、mean8 14.253872。
- hidden-like typewell-purged RMSEはmean4 14.244614、mean8 14.072542。
- 1000+ RMSEはmean4 14.428245、mean8 14.316429。
- mean8-mean4 path差はmean absolute 1.257438 ft、RMSE 2.400394 ft、max 21.104492 ft。
- seed std8とmean8 absolute errorのSpearman相関は0.501475で、target-free disagreementは
  error-risk feature候補になる。ただしhighest-disagreement decileでもmean8が常にmean4を上回るわけではない。

## 再現性

- successful kernel: `kentookumura/exp271-pf-ancc-small-seed-mean-audit-train` version 2
- candidate gzip raw SHA: `a01f2082717c17c5c22ef26dc91f7f87cc98cb48e2d4e0c92dc0a9a0b922590a`
- candidate decompressed SHA: `a7c48204d6782e62941e433b5d47ba5e03f6e441b8e601461be1c63ebcdca336`
- candidate schema SHA: `9037c3e40cd7a4ad8535479dcad7ee16885c2214940a6c357915e8ec8b2a5ba9`
- input manifest SHA: `faea0fdeac2017a4f98456f3f57515442caa2d00e21d2cafd275a394950e2ce4`
- artifact manifest SHA: `3c697683083cb921250dd2e9a00003c85054be5e05087e13ee6db88b92aa7632`
- model / submission SHA: 非該当

ローカル取得した全小型CSV/gzipはmanifestのraw/decompressed SHAと一致した。134,426,437 bytesの
candidate gzipはKaggleに保持し、ローカルへはダウンロードしていない。

## 解釈

仮説は支持する。単一の低コストcandidateを選ぶなら、mean4はmean8単独oracle改善の約93〜95%を
半分のseedで回収し、unique-best率も同等なのでmean4へ縮約する。一方、mean4とmean8を同時に
candidate bankへ入れると追加headroomがあるため、保存済みpathを使う次のtrain-side add-only監査では
両方とseed disagreementを残す。selector gainがmean8へ依存しなければraw-test契約は4 seedに縮約する。

## 次

保存済みmean4/mean8 pathとseed/particle disagreementをexp263/exp264系candidate selectorへ
add-onlyで加えるfold-safe OOF監査を別候補とする。PF再生成、hard oracle routing、raw-test inference、
submissionは、そのsame-run control比較とhidden-like/worst-well guardを通るまで行わない。
