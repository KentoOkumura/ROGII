# 設計

## 目的・仮説

raw/smoothed GRの双方で持続する高不一致ridgeが、正解pathそのものを決めなくてもcorridor間の不可能な越境を表せるかを、固定candidateに対するread-only auditで原因分離する。

## アプローチ

各wellについて、evaluation row×typewell TVT grid上にGR不一致面を作る。horizontal/typewell GRはそれぞれ全well内のmedian/IQRでrobust scaleし、raw系列と事前固定したrolling-median系列の両方がthresholdを超えるセルだけを高不一致候補とする。binary openingでMD方向の持続長とTVT方向の厚みを満たすridgeだけを残す。

高不一致がstate gridの大半を占めるrow、raw GR missing row、局所flat rowはunsupportedとしてbarrierを無効化する。supported rowの青い連続TVT intervalをcorridor nodeとし、隣接supported row間でTVT intervalが重なる、または事前固定した最大corridor shift以内の場合だけedgeを張る。最初のsupported rowで`last_known_tvt`を含む、または最も近いnodeをanchor componentとする。

保存済み`pf_ancc / beam_mean / likpf_mean / sc_ens / hyb` pathについて、各rowのendpointがanchor componentに属するか、連続rowのTVT線分がridgeを横断するかを記録する。true TVTも同じreadoutに通すが、barrier・component作成後の評価専用とする。候補値や予測は変更しない。

## 実験範囲

- 対象実験: `exp246_negative_space_gr_barrier_audit`
- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`（heatmap signal）、`exp072_exp063_full_replay_feature_cache`（固定candidate cache）、`exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission`（worst-well safety evidence）
- 変更する変数: raw/smoothed GR consensusから作るnegative-space ridge、anchor-connected corridor readout、candidate exclusion diagnostic。
- 固定する変数: exp072 candidate値、truth/scoring surface、prediction start、candidate family、no-training、no-inference、no-submit。
- active variant / model / fold / booster: `1 / 0 / 0 / 0`。親/control再学習なし。

## 再現性設計

- seed policy: `no_new_rng_deterministic_barrier_audit`。config互換用seedは42だが乱数処理には使わない。
- stochastic 処理の有無: 新規処理なし。上流exp072 candidate cacheにはstochastic provenanceがあるため、本実験をdeterministic prediction anchorとは呼ばない。
- PF/Beam / likelihood-PF / seed bagging の有無: 再生成なし。保存済みcandidate列をread-onlyで監査する。
- 並列処理と乱数の関係: wellをsorted順に1 processで処理し、RNGなし。thread schedulingに依存する集計を避ける。
- CPU/GPU runtime と deterministic flags: CPU、GPU disabled、internet disabled、num_workers 1。
- train cache / test feature regeneration の SHA 記録方針: exp072 gzip raw SHAとdecompressed content SHA、header/schema、row/well数を記録。raw train file inventory SHAとoutput content SHAを記録。test regenerationは対象外。
- model manifest / prediction / submission SHA 記録方針: model/prediction/submissionなし。barrier summary、row audit、by-well、bucket、hidden-like CSVとconfigのSHAを記録する。
- Kaggle package bootstrap 確認方針: prepare後にCPU/internet off、active variant、kernel sources、config threshold、0 boosterをpackage notebookとmetadataで確認する。

## リスク

- リークリスク: threshold/morphology/component/fallbackをsame-well tail truthで選ぶとnegative spaceがtarget proxyになる。初回設定は事前固定し、true TVT/errorは評価列に限定する。
- CV/LB 不一致リスク: train true pathがred ridgeを横切らなくてもhidden wellのGR scale/noise/typewell relationが異なる可能性がある。hidden-likeとGR missing/flat subgroupを必須にする。
- ランタイム/メモリリスク: 3.78M rows×typewell gridを全well一括materializeするとOOMする。well単位streaming、state cap 256、usecols限定、row-long出力をcompact boolean/float列に限定する。
- 再現性リスク: upstream candidate cacheはstochastic provenanceを持つ。input content SHA固定とsorted well orderで監査差分を追う。
- 科学リスク: hard barrierは1回のfalse cutでtail全体を誤componentへ閉じ込め得る。初回はdiagnostic onlyとし、hard edge-cutはtrue-path cut、candidate precision、1000+、hidden-like、worst-well guard通過後にだけ別段階で検討する。

## 次のアクション

Kaggle CPUでfull 773-well diagnosticを1 variant / 0 config / 0 fold / 0 boosterとして実行する。科学的な採否とhard edge-cut段階への移行可否は、事前固定した5 guardsだけで判断する。
