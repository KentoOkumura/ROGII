# exp102_confidence_gated_likpf_fallback_on_exp101 結果

## 状態

Kaggle train v2 完了。train-side では支持あり。ただし提出候補化はしない。

## 仮説

exp101 の supervised ranker は全体では `likpf_mean` 単体を超えなかったが、信頼度が高い行だけ `pf_ancc` / `beam_mean` へ切り替えれば、低 switch-rate で改善する可能性がある。

## 設定

- 親: `exp101_pf_candidate_ranker_or_nway_classifier`
- 入力: exp099 v2 wide train feature cache
- モデル: exp101 saved LightGBM booster
- default: `likpf_mean`
- 切替候補: `pf_ancc`, `beam_mean`
- 検証: GroupKFold by `well` の OOF posthoc audit

## 判定基準

提出候補化はしない。次へ進む条件は、low switch-rate gate が `likpf_mean` RMSE 11.594898 を改善し、within10、path switch、worst-well を悪化させないこと。改善した場合も continuity / raw-test parity の別実験を必要とする。

## 実装検証

- `py_compile`: 成功
- `ruff check experiments/exp102_confidence_gated_likpf_fallback_on_exp101`: 成功
- `make validate-exp EXP=exp102_confidence_gated_likpf_fallback_on_exp101`: 成功
- Kaggle train package 生成: 成功
- local booster load smoke: 未実行。local `.venv` に `lightgbm` がないため Kaggle runtime で確認する。

## 結果

Kaggle train v2:

- Kernel: `kentookumura/exp102-likpf-fallback-train`
- status: `COMPLETE`
- runtime: 874.76 sec
- rows: 3,783,989
- wells: 773
- output: `kaggle/output/train_v2`

| variant | mode | RMSE | MAE | within10 | switch rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `oracle` | oracle | 7.434030 | 3.745228 | 0.906525 | 0.614770 |
| `gate_error_margin_sr050_d020_std000020` | gated | 11.561206 | 7.053293 | 0.771966 | 0.050000 |
| `likpf_mean_single` | baseline | 11.594898 | 7.067633 | 0.772807 | 0.000000 |
| `exp101_error_ranker_rowwise` | oof | 11.600097 | 7.006912 | 0.771452 | 0.452061 |

best gate は `gate_error_margin_sr050_d020_std000020`。`likpf_mean_single` から RMSE `-0.033692` 改善、MAE `-0.014340` 改善。within10 は `0.772807 -> 0.771966` で `-0.000842` 悪化した。

best gate の selection distribution:

- `likpf_mean`: 3,594,790 rows / 95.000%
- `pf_ancc`: 149,880 rows / 3.961%
- `beam_mean`: 39,319 rows / 1.039%

## 再現性

- exp099 input raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 input decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp101 model manifest SHA: `4f453761f1cc09042767baa934f8a1c5a89036bfb1c244a5f3fc5ab0cc843cc5`
- OOF predictions decompressed SHA: `469e9fa137ffa7f3924711dbe1bb8f67f99d97678442c3d209426031a5f48330`
- best gate prediction SHA: `5181ebf6118fde1f78a5be8ea591fc8d9b05b8c15fa247d7e7bb711b12de8c79`

## 解釈

confidence gate は exp101 row-wise selector の過剰な切替を 45.2% から 5.0% へ抑え、`likpf_mean` default から RMSE を小さく改善した。これは exp101 の ranker confidence が一部の行では有効であることを示す。

一方で within10 はわずかに悪化し、bucket / worst-well でも改善と悪化が混在する。したがって、このまま inference port / submit はしない。次に進めるなら、well 内 continuity penalty、minimum segment length、raw-test feature parity、worst-well guard を加えた診断に限定する。
