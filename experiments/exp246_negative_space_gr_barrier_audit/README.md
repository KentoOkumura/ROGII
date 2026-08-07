# exp246_negative_space_gr_barrier_audit

## 状態

- ルート: PF/Beam (`pf_beam`)
- 状態: Kaggle CPU train v2完了、5 safety guardsすべてfail、hard barrier不採用
- CV: 対象外（no-training diagnostic）
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-14
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe` / `exp072_exp063_full_replay_feature_cache`
- 安全性比較: `exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission`

## 仮説

horizontal row×typewell TVT のGR mismatch heatmapで、raw GRと軽いsmooth GRの双方が高不一致となる持続的なridgeは、正しいpathを引くunaryではなく、青いcorridor間を越境できないnegative-space barrierとして使える可能性がある。候補生成やrank付けよりも、明らかなmode jumpの除外の方が高精度に行えるかを先に調べる。

## 変更点

- exp072の保存済み`pf_ancc / beam_mean / likpf_mean / sc_ens / hyb`をread-onlyで監査する。
- raw horizontal/typewell GRをwell内robust scaleし、raw/smoothed差のconsensusからbarrier候補を作る。
- MD方向の持続長、TVT方向の厚みをbinary openingで要求する。
- GR missing、局所flat、state gridの85%超が高不一致になるrowはunsupportedとしてbarrierを無効化する。
- known-prefix最終TVTをanchorにしたcorridor到達可能性と、candidate pathのendpoint違反・ridge越境を測る。
- 初回はcandidate、HMM/PF/Beam transition、predictionを変更しない。

## 検証方針

- Fold: model foldなし。全773 train wellsのofficial evaluation tailを監査する。
- Group: well単位のby-well、distance bucket、exp115 hidden-like subgroup。
- Stratification: `000_050 / 050_100 / 100_250 / 250_500 / 500_1000 / 1000_plus`。
- Leakage Check: barrier、support、corridor、fallbackにはtail true TVT、target、candidate error、oracleを使わない。true TVTはbarrier完成後のcut/survival評価だけに使う。
- 主指標: true-path瞬時違反率（赤領域への着地・anchor corridor外・ridge越境の和）、anchor-component survival、candidate false-prune、bad-candidate prune precision、union oracle before/after、no-survivor率、worst-well delta。

## 実行入口

- 学習/監査 notebook: `exp246_negative_space_gr_barrier_audit_train.ipynb`
- 推論 notebook: `exp246_negative_space_gr_barrier_audit_inference.ipynb`（diagnostic-only guardで停止）
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp246_negative_space_gr_barrier_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp246-negative-space-gr-barrier-audit-train --title 'exp246 negative space gr barrier audit train' --run-on-push --strict"`
- notebook 実行: Kaggle CPU kernel run を正とする。ローカル実行は`--allow-local`を付けた明示的smoke debugだけに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| 実装 | 完了 |
| active variant | 1 (`diagnostic_only`) |
| LightGBM config / fold / booster | 0 / 0 / 0 |
| 親/control再学習 | なし |
| static validation / strict Kaggle train package | pass / pass |
| Kaggle train-side audit | v1生成物SHA失敗 / v2完走 |
| processed wells / rows | 773 / 3,783,989 |
| runtime | 733.672秒 |
| guard pass | false（0 / 5 pass） |
| union oracle before → fallback | 7.434021 → 9.090919（+1.656898） |
| good candidate false-prune | 0.361574 |
| worst-well delta | +77.747616 ft |
| Public / Private LB | - / - |

## 所見

### 良かった点

- heatmap path generatorを再開せず、negative spaceだけを原因分離した。
- exp072の2GB級gzip cacheを必要列だけchunk読込し、well単位・最大256 stateで処理するため全heatmapを一括保持しない。
- unsupported rowをhard wallにせず、候補0件時もnearest-blueへ投影せず未変更unionへ戻す診断値を分けた。
- synthetic contractで、赤いridge越境、anchor corridor維持、unsupported row neutralを確認するassertionを入れた。
- `likpf_mean`と`pf_ancc`ではbad-candidate prune precisionがbase bad率の1.394倍 / 1.301倍となり、弱いcandidate-risk signal自体は確認できた。

### 悪かった点

- 5 safety guardsはすべてfail。true-path瞬時違反0.006422、component survival 0.991760、good false-prune 0.361574だった。
- union oracleは+1.656898悪化し、635/773 wellsが悪化、worst-wellは+77.747616 ftだった。
- hidden-like 2面でもunion oracleが+2.189781 / +2.204004悪化した。
- upstream exp072 candidate cacheはstochastic provenanceを持つため、本実験をdeterministic prediction anchorとは呼べない。
- v1はgzip下位bufferのclose漏れでSHA段階に失敗し、科学設定を変えないclose修正後のv2で完走した。

### リスク / 注意

- 1回のfalse barrierで以降のpathを無効化するhard-history readoutは意図的に厳しい。true instantaneous violationとhistory後valid率を分けて読む。
- 別corridorがanchor componentから分岐した場合は両方をreachableにし、実際のmode switchはedge-crossingで別に検出する。
- barrier guard不通過のためthreshold gridやHMM/PF edge cutへ進まず、このhard-barrier仮説を閉じる。

## 次

hard edge-cut、raw-test inference、submissionは行わない。signalを残す場合は、累積pruneではなく瞬時endpoint/crossing/barrier fractionをadd-only confidence featureとして既存`topk_path_confidence_features`候補へ統合する。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、実験名や設定名を除いて日本語優先で記録する。
