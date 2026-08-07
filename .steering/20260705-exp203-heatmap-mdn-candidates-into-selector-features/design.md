# 設計

## アプローチ

exp184 の selector training flow を親にして、exp202 の `heatmap_candidates.csv.gz` を読む `hmdn_` feature block を追加する。候補集合そのものは変えず、LightGBM candidate error ranker が「既存候補が heatmap topK と近いか」「heatmap signal が信頼できそうか」を判断できる特徴だけを増やす。

## 実験範囲

- 対象実験: `exp203_heatmap_mdn_candidates_into_selector_features`
- Route: `pf_beam`
- 親実験: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- 変更する変数: exp202 heatmap MDN topK 由来の `hmdn_` row / candidate-long features
- 固定する変数: selector 候補集合、exp184 `hmpf_` features、GroupKFold、LightGBM config、Viterbi grid

## 再現性設計

- seed policy: exp184 と同じ GroupKFold seed、LightGBM seed、candidate-long row subsample seed を使う。
- stochastic 処理の有無: 本実験の feature generation は deterministic。upstream exp182/exp202 は PyTorch CUDA artifact として stochastic source に記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: exp072/exp099 の保存済み train-side cache を読む。再生成はしない。
- 並列処理と乱数の関係: feature generation は no RNG。LightGBM と candidate-long subsample のみ fixed seed。
- CPU/GPU runtime と deterministic flags: Kaggle CPU train、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train 実行後に model manifest、OOF prediction、feature schema、summary SHA を記録。submission は生成しない。
- Kaggle package bootstrap 確認方針: prepare-kaggle-notebooks 時に exp099/exp072/exp182/exp202/exp115 source を含むことを確認する。

## リスク

- リークリスク: exp202 CSV には true center / abs-error 系列が含まれるため、feature usecols を allowlist に固定する。
- CV/LB 不一致リスク: selector train-side positive でも raw-test heatmap generation parity が未確認なら inference に進めない。
- ランタイム/メモリリスク: candidate-long feature が増えるため、exp184 と同じ 120k row/fold cap と chunk prediction を維持する。
- 再現性リスク: upstream exp202 が GPU diagnostic artifact なので deterministic submission anchor にはしない。
