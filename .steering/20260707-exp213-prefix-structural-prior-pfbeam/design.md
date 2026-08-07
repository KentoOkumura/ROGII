# 設計

## アプローチ

exp211 の train-side PF/Beam audit surface を再利用し、affine GR observation 軸を外して P0-B の raw GR + prefix structural prior に集中する。structural prior は known prefix tail の `TVT_input + Z` を MD に対して robust linear fit し、evaluation rows の `Z` を引いて expected TVT を作る。

PF では expected TVT を hard window にせず、初期速度の blend、各 step の velocity pull、absolute TVT soft likelihood として使う。Beam では start index を last known `TVT_input` に固定し、absolute TVT soft cost と expected step-delta costを既存の GR/move cost に足す。top-K path は row candidates に残す。

## 実験範囲

- 対象実験: `exp213_prefix_structural_prior_pfbeam`
- Route: `pf_beam`
- 親実験: backlog `prefix_structural_prior_pfbeam`
- 実装親: `exp211_affine_calibrated_gr_observation_pfbeam`
- 変更する変数: structural prior の有無、absolute position weight、step-delta weight、velocity blend / pull
- 固定する変数: raw GR observation、target wells、PF particles/seeds、Beam size/move radius、exp072 reference candidate surface

## 再現性設計

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_structural_variants`
- stochastic 処理の有無: PF particle propagation / resampling あり
- PF/Beam / likelihood-PF / seed bagging の有無: PF と Beam の train-side generation audit のみ。LightGBM / booster / submission はなし
- 並列処理と乱数の関係: 初回実装は sequential。各 well / seed index ごとに stable seed を作り、global RNG は使わない
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled、internet disabled
- train cache / test feature regeneration の SHA 記録方針: Kaggle train output 取得後、input cache SHA と row candidates gzip の decompressed content SHA を主証拠として記録する
- model manifest / prediction / submission SHA 記録方針: model / submission は生成しない
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に generated package の metadata と bootstrap support files を検証する

## リスク

- リークリスク: structural fit に evaluation true TVT を混ぜると即リークになる。fit は known prefix `TVT_input`、MD、Z のみに限定する
- CV/LB 不一致リスク: train-side pseudo-tail で positive でも raw hidden test regeneration、worst-well、near rows、longtail が未確認なら inference/submit しない
- ランタイム/メモリリスク: 64 wells x 4 variants x 240 particles x 8 seeds + Beam top-K で exp211 より少し重い。Kaggle CPU train の範囲で実行する
- 再現性リスク: PF の resampling は stochastic。variant 間の seed を共有し、raw vs structural 差分が RNG 消費差ではなく prior 差分として読めるようにする
