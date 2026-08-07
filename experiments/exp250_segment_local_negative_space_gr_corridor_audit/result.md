# exp250_segment_local_negative_space_gr_corridor_audit 結果

## 状態

Kaggle CPU Stage 0 / Stage 1を完了した。Stage 0 manual parityはPASS、Stage 1の事前固定guardは2/8 PASS、6/8 FAILだった。判定は`fail_close_segment_local_hard_use_and_grid_search`であり、このsegment-local corridorをhard利用、parameter grid、raw-test inference、submissionへ進めない。

## 仮説

segment内だけでresetするminimum-bottleneck GR corridorなら、exp246のfull-tail hard-history barrierより低いgood-candidate誤警報でbad candidate rowを濃縮できる可能性がある。

## 固定設定

- Route: `pf_beam`
- 親: `exp246_negative_space_gr_barrier_audit`
- segment: MD 256 ft / stride 128 ft / horizontal 4 ft/bin
- typewell: flat-Z prior ±256 ft / 4 ft/state / 129 states
- normalization: well-wide separate robust-z、clip [-8, 8]
- surfaces: `real_gr`, `shuffled_typewell_gr`
- corridor: minimum-bottleneck `tau_star + 0.25`
- candidate / model config / fold / booster: 5 fixed / 0 / 0 / 0
- candidate変更、親control再学習、raw-test推論、提出: なし

## 実行結果

773 wells / 3,783,989 candidate rowsを処理した。Stage 1 audit runtimeは7,633.823秒、notebook全体は約7,898.7秒だった。primaryの評価row weightは3,652,581、candidate-segment sampleは145,855だった。

| Primary metric | 値 |
| --- | ---: |
| real GR pooled bad-candidate AUC | 0.530134 |
| shuffled GR pooled AUC | 0.494199 |
| real - shuffled AUC | +0.035934 |
| q90 risk threshold | 1.000000 |
| q90 bad-rate lift | 0.776971x |
| q90 good-candidate false-alert | 0.232020 |
| truth coverage real - shuffled | +0.059580 |

q90 riskが1.0へ飽和し、high-risk側のbad rateは母集団より低かった。GR topology固有の差は存在するが、candidate誤りのglobalな識別器としては弱い。

## Guard判定

| Guard | 観測値 | 判定 |
| --- | --- | --- |
| pooled real AUC >= 0.60 | 0.530134 | FAIL |
| real - shuffled AUC >= 0.02 | +0.035934 | PASS |
| `likpf_mean` / `pf_ancc` q90 liftとcontrol差 | real lift 0.997198 / 1.078168、control差 -0.017351 / +0.049454 | FAIL |
| q90 good false-alert | overall 0.232020、family max 0.262491 | FAIL |
| overlap path / risk agreement | path median 57.61 ft、p90 258.684 ft、Spearman 0.448723 | FAIL |
| hidden-like AUC | spatial 0.531044、typewell-purged 0.532323 | FAIL |
| by-well good false-alert | p95 0.757381、max 0.984733 | FAIL |
| truth coverage real - shuffled | overall +0.059580、1000+ +0.061222、hidden-like +0.047406 / +0.048384 | PASS |

## Candidate family

| Candidate | real AUC | real q90 lift | good false-alert |
| --- | ---: | ---: | ---: |
| `beam_mean` | 0.516251 | 1.045051 | 0.256437 |
| `hyb` | 0.524232 | 1.001447 | 0.100100 |
| `likpf_mean` | 0.507163 | 0.997198 | 0.262491 |
| `pf_ancc` | 0.518633 | 1.078168 | 0.252687 |
| `sc_ens` | 0.521765 | 1.008537 | 0.095876 |

どのfamilyもAUCとq90 liftの両方で利用水準に届かなかった。

## 距離別の解釈

0–50 ft / 50–100 ftではreal AUCが0.820631 / 0.819121、shuffled AUCが0.775910 / 0.771389、q90 liftが2.042545 / 2.002634だった。一方、支配的な1000+ bucketは評価weight 2,950,940でreal AUC 0.515575、q90 lift 0.785417だった。

near bucketには局所signalが見えるが、shuffled control自体もAUC約0.77であり、距離とbase error rateの構造が大きく混ざる。これをglobal corridor featureの採用根拠にはしない。

well間の不安定性も大きく、worst `d07aed8f`はAUC 0.005368、good false-alert 0.984733だった。overlapしたsegment間のpath差とrisk相関もguardから大幅に外れたため、segment境界に対して安定なsignalとはいえない。

## 結論

局所GR topologyにはshuffled controlを上回る弱い情報があるものの、誤警報、overlap安定性、hidden-like、worst-wellの全てで利用条件を満たさなかった。契約どおり次を閉じる。

- candidateのhard prune / replacement / edge cut
- threshold、slack、segment、stride、half-widthの事後grid
- `topk_path_confidence_features`やnested selectorへのsegment-local signal追加
- raw-test inference、submission

失敗原因の切り分けを再訪する場合だけ、保存済みcandidate-segment artifactを使い、near bucketの見かけのAUCがdistance-conditioned base rateで説明されるかを読む低優先のattribution readoutとする。新しいcorridor計算やparameter探索は行わない。

