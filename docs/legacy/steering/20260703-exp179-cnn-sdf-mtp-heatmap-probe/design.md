# 設計

## アプローチ

discussion 699853 の CNN+SDF+MTP を、提出モデルではなく fold-safe な GPU diagnostic として最小化する。horizontal well の疑似 hidden tail から row window を作り、typewell GR/TVT window と組み合わせて 5ch heatmap を作る。CNN は `K=10` の path head を出し、各 mode が horizontal window 各 row の typewell TVT bin を分類する。loss は mode ごとの row-wise CE のうち最小 mode を選ぶ closest-mode loss と mode-logit CE を併用する。

window center は true TVT ではなく、観測済み prefix の最後の `TVT_input` と full horizontal `Z` から作る flat prior `last_known_tvt - (Z - last_known_z)` に固定する。これにより valid true TVT は label / metric のみに閉じる。入力はまず discussion 準拠の 5ch に限定し、sin/cos dip や location prior は後続 ablation に回す。

## 実験範囲

- 対象実験: `exp179_cnn_sdf_mtp_heatmap_probe`
- Route: `ml_model`
- 親実験: `mtp_heatmap_sdf_mdn_probe` backlog / discussion 699853
- 変更する変数: 5ch heatmap CNN/MTP probe の新規実装、GR control variant (`real_gr`, `shuffled_gr`, `no_gr`)
- 固定する変数: raw train data、well GroupKFold、target-free flat prior window center、sample schedule、epochs、K path modes、window shape
- 実行しないもの: PF/Beam replacement、current-test inference、submission、既存 ML/PF baseline control 再学習

## 再現性設計

- seed policy: global seed 42 を固定し、well sample order と shuffled-GR roll は SHA256 keyed stable seed で決める。
- stochastic 処理の有無: PyTorch CUDA conv training、AdamW、DataLoader shuffle が stochastic。`torch.use_deterministic_algorithms(True, warn_only=True)`、CuDNN deterministic true、benchmark false を設定する。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: DataLoader `num_workers=0`。thread scheduling に依存する RNG は使わない。
- CPU/GPU runtime と deterministic flags: Kaggle GPU 必須。CUDA がない場合は RuntimeError。CPU fallback はしない。
- train cache / test feature regeneration の SHA 記録方針: sample index、validation predictions、feature schema は CSV/CSV.GZ で保存し、gzip は decompressed content SHA を summary に記録する。test feature regeneration は範囲外。
- model manifest / prediction / submission SHA 記録方針: variant ごとに `state_dict` を保存し SHA を manifest に記録。validation prediction gzip の raw/decompressed SHA を記録。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --run-on-push --strict` 後、metadata の `enable_gpu=true` と bootstrap 内 `config.yaml` の一致を確認する。

## リスク

- リークリスク: true TVT で typewell window を中心決めすると簡単に漏れるため禁止。valid true TVT は label / metric のみ。history channel は `TVT_input` prefix の finite 値だけを使う。
- CV/LB 不一致リスク: train-side diagnostic であり、LB や submit 候補ではない。positive でも full-length inference port 前に negative control、worst-well、raw/current-test parity が必要。
- ランタイム/メモリリスク: 5ch 128x64 image、small wells、1 fold、3 variants、5 epochs に制限する。GPU memory は batch size 32 で抑える。
- 再現性リスク: GPU training は bitwise anchor としない。seed、model SHA、sample index SHA、prediction SHA、Kaggle version を記録する。
