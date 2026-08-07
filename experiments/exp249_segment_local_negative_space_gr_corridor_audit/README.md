# exp249_segment_local_negative_space_gr_corridor_audit

## 状態

- Route: `pf_beam`
- 状態: Kaggle Stage 1 COMPLETE、guard failed、不採用
- CV / Public LB / Private LB: なし
- 推論 / 提出: 無効

## 仮説

exp246のwell全tail global hard-history barrierは誤pruneが大きかったが、着想元と同じ局所GR mismatch表示に限定し、segmentごとにcorridor/historyをresetすれば、赤いridge越境eventがbad candidate riskを誤警報の少ない形で濃縮できる可能性がある。

## 変更点

- horizontal 128 rows × typewell 64 bins、target-free prior ±192 ft。
- stride 64のoverlapping segment。
- horizontal segmentとtypewell cropを別々にmedian/IQR scale。
- barrier/component/historyはsegment内でreset。
- overlap viewは統合せず、agreementとinverse-coverage weighted rateを保存。
- exp072固定candidateを変更せず、学習・prune・平均・inference・submitを行わない。

## 実行段階

1. `stage0_preview`: 少数wellのsigned/absolute mismatch、barrier、truth/candidate overlayを保存し、pixel/axis/normalization parityを確認する。
2. `stage1_full_audit`: parity確認後だけ773 wellsのrisk enrichment / false-alert auditを実行する。

Stage 0 Version 1でmanual parityを確認し、Stage 1 Version 2で773 wellsを監査済み。

## 検証方針

Stage 0は少数wellのPNGとpixel/axis metadataを保存し、着想元との表示parityを手動確認する。Stage 1は固定candidateに対する773-well train-side auditで、bad-candidate precision lift、good-candidate false-alert、truth survival、overlap disagreement、boundary感度、1000+、hidden-like、by-well worstを評価する。target/error/oracleはsignal固定後の採点にだけ使い、windowやthresholdの選択には使わない。

## 所見

Stage 0では3 wells × 3 positionsの9画像を生成し、128×64、±192 ft、axis、色範囲、local normalization、端部clipがexp202/208契約と一致した。一方、Stage 1はbad-candidate precision lift 0.917x、good false-alert 0.541、truth false-alert 0.537でguard不通過だった。局所化してもcomponent transitionがtruthとgood candidateを約半数flagし、risk濃縮にはならなかったため不採用とする。

## 参照

- 親: `exp246_negative_space_gr_barrier_audit`
- 局所window: `exp202_heatmap_mdn_candidate_generator_probe`
- stride: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- fixed candidates: `exp072_exp063_full_replay_feature_cache`
