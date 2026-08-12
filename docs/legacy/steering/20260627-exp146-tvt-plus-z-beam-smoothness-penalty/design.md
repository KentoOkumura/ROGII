# 設計

## アプローチ

exp072 の Beam search 実装を基準に、GR matching cost と TVT move penalty は維持する。新しい Beam variant では各遷移で次を追加する。

- `U = TVT + Z - (T0 + Z0)` の絶対値 penalty。
- `dU/dMD = dTVT/dMD + dZ/dMD` の slope penalty。
- 任意で `dU/dMD` の step 間変化 penalty。

これにより、坑跡 Z の変化と TVT の変化を合わせた trajectory consistency を Beam path selection の中で直接扱う。固定済み `beam_mean` への後処理補正ではなく、Beam の探索 cost を変える。

## 実験範囲

- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 比較対象: exp072 `likpf_mean`、`beam_mean`、`pf_ancc`、`pf_z`、再生成 Beam replay、TVT+Z penalty Beam variants。
- score rows: exp072 feature cache と同じ train pseudo-tail rows。

## 再現性

- 新規 RNG は使わない。
- raw horizontal/typewell file SHA、exp072 cache SHA、生成 candidate wide SHA、metrics SHA を summary JSON に記録する。
- notebook 初回実行は Kaggle train notebook とする。

## リスク

- `U` への過剰追従で near-prefix や low-Z-change wells を壊す可能性がある。
- Beam replay が exp072 `beam_mean` と完全一致しない可能性があるため、replay variants は比較用 control として扱う。
- train pseudo-tail で改善しても hidden raw-test に再現しない可能性があるため、改善しても即 submit しない。
