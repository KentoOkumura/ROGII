# exp179_cnn_sdf_mtp_heatmap_probe 結果

## 仮説

discussion 699853 の 5ch heatmap CNN/SDF/MTP が、target-free window 内で真値近傍の trajectory mode を topK に残せるか確認する。

## 設定

- 親: `mtp_heatmap_sdf_mdn_probe` backlog / discussion 699853
- 検証: 1 fold well GroupKFold heatmap MTP probe
- メトリック: top3 within10 center coverage
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| top3 within10 (`real_gr`) | 0.44921875 |
| top3 within10 (`shuffled_gr`) | 0.232421875 |
| top3 within10 (`no_gr`) | 0.0625 |
| top10 within10 (`real_gr`) | 0.794921875 |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false。GPU train-side diagnostic として扱う。
- seed policy: fixed global seed + well keyed SHA256 sample/control seeds
- kernel version: `kentookumura/exp179-cnn-sdf-mtp-heatmap-probe-train` v2
- runtime: Tesla T4 / PyTorch 2.10.0+cu128 / internet disabled
- sample index decompressed SHA: `4c23b5ca13a872cf6fd085f3d2c225357e2a73ecb5b47cf6ce423211127bfb92`
- feature schema SHA: `a2c4fda671361eab9e876ff84e3c4600ac4d6cec1727aad1035dfd623b62e367`
- model manifest SHA: `9870e05c6de6b244bdf75f66c01b3b405ea0a53ed08f38e7b1fa067feeebdef6`
- model SHA: `real_gr=c65ac7f5ac9f04d8b41477e0cff0fc79ca9b31f1fdcc35bde4a7b367eb9a9fb5`, `shuffled_gr=3729ef2325795af6a1f7b1362bf6341b575ad46232948f6c4b25f282f954fbc9`, `no_gr=1d322e3e283194fe11951c3521bed1c22f52c95e94c29234a41c054a37bb2e7d`
- validation predictions decompressed SHA: `2befa525c5922d3ac1cda7a38fd23e134e48b2618176e98b0c7bf4343a08d7ca`
- summary SHA: `04efe7b405bb8d090b72626f7af5961235cceda52834c5a7c76b66863a9a225d`
- submission SHA: なし
- rerun result: 未実行

## 解釈

v1 は GPU runtime で起動したが、Kaggle 側の割当が P100 だった。現在の Kaggle PyTorch 2.10 は P100 の CUDA capability `sm_60` をサポートしておらず、モデルを CUDA に載せる時点で `no kernel image is available for execution on the device` により失敗した。

v2 は T4 明示で完了した。`real_gr` は top3 within10 0.449219 で、`shuffled_gr` 0.232422 と `no_gr` 0.062500 を明確に上回った。top10 within10 も `real_gr` 0.794922、`shuffled_gr` 0.541016、`no_gr` 0.062500。したがって 5ch heatmap CNN は少なくともこの target-free window / 1 fold smoke では GR signal を使えている。

一方、これは 1 fold / 160 selected wells / 128x64 fixed window の smoke であり、full-length inference や提出に進める根拠ではない。次は full-fold confirmation、larger window、geometry channel ablation、worst-well / distance bucket readout を見る。

## 次

`cnn_sdf_mtp_heatmap_probe` は完了。次は direct replacement ではなく、full-fold / larger-window / geometry-channel ablation として継続可否を判断する。
