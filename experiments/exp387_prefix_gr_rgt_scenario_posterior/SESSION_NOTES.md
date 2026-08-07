# exp387_prefix_gr_rgt_scenario_posterior セッションノート

## 目的

exp386 の固定 scenario bank に対し、既知 TVT prefix と target GR の尤度で posterior を更新し、物理 scenario の posterior mean が exp226 と prior-only mean を大幅に上回るか検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: exp386 Stage 0 FAILにより未実装で閉鎖
- CV / LB: なし
- 設計承認: あり
- 実装 / notebook 採用 / package / push / run / inference / submission: 未承認

## 2026-07-24 親gate不成立による閉鎖

- exp386 version 1（id_no `128478384`）は`COMPLETE`したがStage 0 FAIL_CLOSE。
- exp386 scenario-bank well coverage、scenario count p05、finite-path coverageはすべて0。
- exp386 cycle residual p95は`2.363303 > 0.10`。
- exp387が必要とする8〜32 scenario / wellとreference-GR templateは生成されなかった。
- 親の全Stage PASSとmanifest SHA pinという開始条件が成立しないため、
  compact実装、正規Notebook採用、package、push、run、inference、submissionなしで閉じる。
- exp387 strict experiment validationとJSON/YAML parseはPASSした。

## 2026-07-24 設計セッション

実行済み:

```bash
make new-steering EXP=exp387_prefix_gr_rgt_scenario_posterior
make new-exp EXP=exp387_prefix_gr_rgt_scenario_posterior
```

設計確定内容:

- exp386 の全 gate PASS と manifest logical SHA 固定を開始条件とする。
- exp386 の scenario 値・順序・prior cost・参照 GR template は変更しない。
- GR level と first difference の固定 Student-t 尤度を使用する。
- graph-cost prior と exact forward-backward を組み合わせ、posterior mean を出力する。
- real GR と 512行 circular-shift GR の識別を Stage 0 の必須 gate とする。
- Stage 1 の主閾値は pooled RMSE 7.20 ft以下かつ exp226 比2.0 ft以上改善とする。

## 予定計算量

- variant: 1
- Stage 0 likelihood audit: 1
- Stage 1 exact decoder well run: 773
- fitted model / LightGBM / HMM / PF / Beam: 0
- exp386 scenario / exp226 control 再生成: なし

## 再現性メモ

- seed policy: RNG を使わず、fold・well・scenario・window・state の不変キーで安定順序化
- stochastic components: なし
- parent SHA: exp386 完了後に `config.yaml` の null placeholder を固定値へ置換
- logical SHA: real/circular score、transition manifest、posterior、OOF prediction に記録予定
- deterministic anchor: 未確立。将来の同一設定 rerun で content SHA 一致を確認してから確立

## 次のアクション

1. 現在のexp386依存設計は再開しない。
2. 別generatorで非空bankと全gate PASSの独立根拠が得られた場合だけ、新しい事前設計を検討する。
