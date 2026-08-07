# exp246_negative_space_gr_barrier_audit 結果

## 結論

negative-space ridgeをhard-history barrierとして候補除外に使う仮説は不採用。Kaggle CPU v2は773 wells / 3,783,989 rowsを完走したが、事前固定した5 safety guardsをすべて下回った。threshold grid、HMM/PF/Beam edge cut、raw-test inference、submissionへは進めない。

## 仮説

GR mismatch heatmapの高不一致ridgeをcandidateを引き寄せるunaryではなく越境不能候補のnegative-space evidenceとして扱うと、正しいpathをほぼ切らずにmode jump候補を除外できる可能性がある。

## 設定

- 親: `exp202_heatmap_mdn_candidate_generator_probe` / `exp072_exp063_full_replay_feature_cache`
- Route: `pf_beam`
- 検証: 773 train wells official evaluation tailのno-training diagnostic
- active variant: `diagnostic_only` 1本
- LightGBM config / fold / booster: `0 / 0 / 0`
- 親/control再学習: なし
- kernel: `kentookumura/exp246-negative-space-gr-barrier-audit-train` version 2 / id_no `127059485`
- runtime: 733.672秒、CPU、internet disabled
- シード: 42（新規RNGなし）

## 実行履歴

- v1: 773-well本体処理後、caller-owned gzip file buffer未closeによりdecompressed SHA読取でEOFError。
- v2: `raw.flush()` / `raw.close()`とhash前closed assertionだけを追加し、科学設定を変えず完走。
- output取得はmetrics、summary、group/candidate/by-well/barrier-well集計だけに限定し、3.78M-row auditはダウンロードしていない。

## Safety guards

| Guard | 値 | 上限 / 下限 | 判定 |
| --- | ---: | ---: | --- |
| true-path瞬時違反率 | 0.006422 | ≤ 0.001 | fail |
| anchor component survival | 0.991760 | ≥ 0.995 | fail |
| good-candidate false-prune率 | 0.361574 | ≤ 0.001 | fail |
| union oracle RMSE delta | +1.656898 ft | ≤ 0.0 | fail |
| worst-well union oracle delta | +77.747616 ft | ≤ 0.25 | fail |

補助指標はbarrier supported率0.677376、true endpoint forbidden率0.005290、true edge crossing率0.004304、no-survivor率0.278630。union oracleは7.434021から、strict survivor 8.925525、no-survivor時の未変更fallback込みでも9.090919へ悪化した。

## Subgroup / by-well

| Group | true-path瞬時違反率 | component survival | no-survivor率 | union oracle delta |
| --- | ---: | ---: | ---: | ---: |
| hidden-like spatial | 0.010500 | 0.985336 | 0.339365 | +2.189781 |
| hidden-like typewell-purged | 0.010491 | 0.985210 | 0.337752 | +2.204004 |

well単位は改善0 / 同値138 / 悪化635。worstは`d07aed8f`で、union oracle 1.938372から79.685988へ悪化し、deltaは+77.747616 ftだった。

## Candidate readout

| Candidate | base bad率 | prune precision | precision lift | good false-prune率 |
| --- | ---: | ---: | ---: | ---: |
| beam_mean | 0.408379 | 0.498448 | 1.221x | 0.302069 |
| likpf_mean | 0.227207 | 0.316701 | 1.394x | 0.312360 |
| pf_ancc | 0.308259 | 0.401136 | 1.301x | 0.283587 |
| hyb | 0.832896 | 0.841735 | 1.011x | 0.760074 |
| sc_ens | 0.864935 | 0.869957 | 1.006x | 0.810191 |

`likpf_mean` / `pf_ancc`ではbad-candidate enrichment自体はあるが、1回の違反をtail全体へ累積するhard-history ruleによりgood candidateも約28–31%切る。hard exclusionとして許容できない。

## 再現性

- deterministic anchor: false。auditは固定入力に対してdeterministicだが、upstream exp072 candidate cacheのstochastic provenanceを引き継ぐ。
- candidate cache decompressed SHA: `99a3c70a...0e1350`
- raw file inventory SHA: `335b336b...1fd9d`
- row audit decompressed SHA: `07d2c918...3f1597`
- group / candidate / by-well SHA: `23e2dd44...3089` / `eecedf39...42f` / `2d6329b0...c9e1`
- model / prediction / submission: 生成なし。
- 実行時config SHAは`4a0e7673...12724`。実行後、local configのstatusだけを`completed_guard_failed`へ更新した。

## 解釈

高不一致ridgeは一部の悪いPF/Beam候補に濃縮して現れるが、「一度越えたら以降すべて無効」というtopological hard barrierには精度が足りない。true path自身も0.642%のrowで違反し、hidden-likeでは約1.05%まで悪化する。ユーザー案のような直接禁止はworst-well破壊が大きく、exp231と同様にhard geometry/GR evidenceの過信が主因と考える。

なおdistance bucket labelは実行configの未引用YAML値が数値化され、出力上`40 / 20544 / 100250 / 250500 / 5001000 / 1000_plus`となった。edgesと集計行は正しく、順に`000_050 / 050_100 / 100_250 / 250_500 / 500_1000 / 1000_plus`へ対応する。科学判定には影響しないため再実行しない。

## 次

negative-space hard barrierはclosedとする。残す場合も、累積pruneではなくendpoint/crossing/barrier fractionを瞬時のadd-only confidence featureとして`topk_path_confidence_features`候補に統合するだけに限定し、direct edge cutやthreshold gridは行わない。
