# exp270 exact HMM posterior mode candidate audit 結果

## 結論

Kaggle CPU aggregate version 4まで完了し、technical contractはすべてPASSした。一方、direct candidateはposterior meanが最良で、marginal MAP、global Viterbi、top-K pathはすべて悪化した。したがってposterior mode pathの直接採用、selector学習、raw-test inference、submissionには進まず、このbranchをnegative resultとして閉じる。

教師値を使うoracleには大きな見かけ上のheadroomがあるが、deployableな選択信号を示すものではない。さらにtop-2からtop-5がmean / MAP / Viterbi bankへ追加するoracle改善は最大でも0.000342 ftに留まり、追加joint path rankの価値は実質的に確認できなかった。

## 仮説

exp209 posterior meanが多峰posteriorの非物理的な中間pathを作る場合、同じHMMのmarginal MAP / global Viterbi / top-K mode pathにはdirectまたはtarget-freeに選択可能なcandidate headroomがある。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- HMM変更: なし
- decoder: joint exact top-5、TVT grid-index sequence dedup、backfillなし
- oracle block: 128 / 256 / 512 rows
- 評価面: full train unknown suffix、distance、hidden-like、by-well、focus `11d0f5ac`
- shard生成量: HMM variant 1、773 well-runs、LightGBM config / fold / booster 0 / 0 / 0
- aggregate量: HMM variant / LightGBM config / fold / booster 0 / 0 / 0 / 0
- seed: 乱数なし

## Direct candidate

| Candidate | Coverage | RMSE | posterior mean差 | 改善well / available | worst-well差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| posterior mean | 1.000000 | 11.938287 | 0.000000 | 0 / 773 | 0.000000 |
| marginal MAP | 1.000000 | 12.592479 | +0.654192 | 206 / 773 | +13.196777 |
| global Viterbi (`topk_path_1`) | 1.000000 | 15.551665 | +3.613377 | 205 / 773 | +55.179043 |
| `topk_path_2` | 0.840600 | 15.302589 | +3.364301 | 175 / 655 | +55.179097 |
| `topk_path_3` | 0.782772 | 15.157698 | +3.219411 | 164 / 613 | +55.179043 |
| `topk_path_4` | 0.759698 | 15.140188 | +3.201901 | 159 / 594 | +55.179097 |
| `topk_path_5` | 0.743562 | 14.674071 | +2.735784 | 159 / 581 | +55.179043 |

posterior meanはRMSEだけでなくMAE 6.769555、within 10 ft 0.784387でも最良だった。marginal MAPのwell別RMSE差中央値は+0.210160 ft、global Viterbiは+3.467042 ftで、改善wellが存在しても全体として安全なreplacementではない。global Viterbiの最大悪化はwell `86454a6f`で、posterior mean 5.098001に対し60.277044、差+55.179043 ftだった。

## Hidden-like

| Subgroup | posterior mean | marginal MAP | global Viterbi |
| --- | ---: | ---: | ---: |
| verification-like spatial | 12.564491 | 13.406237 | 16.989852 |
| verification-like typewell-purged | 12.367244 | 13.217184 | 16.699334 |

hidden-like 2面でもposterior meanが最良で、mode pathへの切替を支持する分布外寄りの証拠はなかった。

## Oracle diagnostic

以下はtrue TVTで候補を選ぶ診断専用値であり、CV候補、selector性能、提出可能スコアではない。oracle predictionは保存していない。

| Scope | all 7 modes | mean / MAP / Viterbi | paths only |
| --- | ---: | ---: | ---: |
| row | 7.516850 | 7.517189 | 15.549160 |
| block 128 | 7.567530 | 7.567871 | 15.549169 |
| block 256 | 7.608996 | 7.609328 | 15.549184 |
| block 512 | 7.685922 | 7.686242 | 15.549194 |
| whole well | 8.536362 | 8.536605 | 15.549395 |

mean / MAP / Viterbi間にはtarget-side oracle headroomがある。一方、all 7 modesと3候補bankの差はrow 0.000340、block 128 0.000342、whole-well 0.000243 ftだけで、top-2からtop-5はほぼ追加価値を持たない。paths-only oracleも約15.549でposterior meanより大幅に悪い。

unique-bestはrow単位でposterior mean 37.5834%、marginal MAP 33.8878%、global Viterbi 4.0647%、well単位で410 / 166 / 61 wellsだった。ただしこれは教師値による事後診断であり、target-free selectorの根拠にはしない。

focus well `11d0f5ac`ではposterior mean 21.160939、marginal MAP 21.218470、global Viterbi 20.793050、row oracle 20.238791、block-128 oracle 20.267534だった。局所改善はあるが、globalなdirect悪化を覆す大きさではない。

## 実装・成果物監査

- aggregate kernel: `kentookumura/exp270-exact-hmm-posterior-mode-audit-train` version 4 / id_no `127594551`
- status / runtime / peak RSS: `COMPLETE` / 156.241秒 / 3,097.277 MB
- rows / wells: 3,783,989 / 773
- exp209 posterior mean parity: max / mean差0.0 / 0.0 ft、tolerance `1e-5 ft`、PASS
- unique path count: 平均4.160414、最小1、5本未満192 wells。no-backfill contractどおり
- ID=`<well>_<row_idx>`、strict well/row順、full-coverage 3候補finite、oracle/selector/blend列不在: PASS
- candidate gzip bytes: 134,101,285
- candidate raw gzip SHA: `bef798c79e902faf93bf8ef5e75c6f868722fc8c4bca61c6dfd10778cfb4d520`
- candidate decompressed SHA: `986276e2495b23d5de2425542efd0433c59f1a398c073b3053218ba9007a5ecf`
- prediction content SHA: `d42a492bc4065ac88b481d9dba1e24b4c3b4ab331c93fce73a433d592aa16f19`
- decoder manifest SHA: `863ccb7ec073e2b624fb59d43ed91a1604cf6a7d17e7a345ab9ccec23a9b9e52`
- candidate schema SHA: `8c75d1fd4e71e81496f6384ea49787dba1dff16329bf0048dc8c65019798006d`
- input manifest SHA: `6ee0a89c40bd505631ce431d7f4f8452576da00fca266af31565a3a333de3a41`
- downloaded `metrics.json` SHA: `1332971e26c5967dc19f57a908d0a888939426fda0f72492604f30b17272bdc5`
- aggregate summary SHA: `5a467f283204809e67851df291c8c1b979d8a767b60b6588cd35ea7a54cb06fa`
- oracle prediction / selector / inference / submission: すべて未生成

small-trellis全列挙、TVT sequence dedup、block oracle、truth後付け、forward-backward smoke、Jupytext、py_compile、Ruff、targeted tests、strict validatorは実行前にPASS済み。取得aggregateではsummaryに記録された13 artifact SHAもすべて実ファイルと一致した。

## 解釈

execution失敗ではなく、科学仮説のdirect側がnegativeである。posterior meanはmode間平均という弱点を持ち得るが、hardなmarginal MAPやjoint pathへ置き換えるとrareな大幅offsetを増やし、overall、hidden-like、well-level safetyをすべて悪化させた。

oracleの7.52--8.54 ftは候補間の相補性を示すが、true TVTを使った上限である。exp270はtarget-free選択性能を測っておらず、oracleだけを理由にselector experimentを作るのは既存のselector失敗履歴とも整合しない。top-2からtop-5の追加価値もほぼゼロなので、少なくともtop-K mode bankを拡張する方向は閉じる。

## 次

exp270固有の救済backlogは追加しない。候補を再利用する場合も、保存済みposterior mean / MAP / Viterbiとposterior mass・gapだけを対象に、別の独立証拠が得られた後のtarget-free readoutから始める。現時点では既存の高優先実験を維持し、exp270 candidate追加、selector、raw-test inference、submissionには進まない。
