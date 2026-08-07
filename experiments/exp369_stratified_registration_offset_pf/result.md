# exp369_stratified_registration_offset_pf 結果

## 仮説

bounded delta層を維持すれば、PFがGR registration modeを早期に失わず追跡できる。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- 検証: known-prefix Stage 0、条件付きStage 1 likelihood-PF
- メトリック: held-out GR NLL、RMSE、delta mass、tail safety
- シード: stable SHA256 per well / seed

## 結果

未実装・未実行。

## 再現性

- deterministic anchor: no
- seed policy: stable SHA256 local RNG
- kernel / feature / prediction SHA: 未生成
- submission SHA: 提出無効

## 解釈

設計確定のみ。実験結果はまだ存在しない。

## 次

実装は別承認まで行わない。
