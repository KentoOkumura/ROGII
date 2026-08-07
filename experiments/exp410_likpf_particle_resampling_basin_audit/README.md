# exp410_likpf_particle_resampling_basin_audit

## 状態

- ルート: pf_beam
- 状態: full 496-well Kaggle CPU audit実行中
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-26
- 親実験: exp072_exp063_full_replay_feature_cache

## 仮説

exp072 likelihood-PFの長いvertical offsetは、HMMと同じtransition / prior
hysteresisだけではなく、PF固有のparticle support、GR emission、ESS resampling、
within-seed particle mean、128 seed arithmetic meanのいずれかで形成されている。

## 変更点

- exp072 exact PFを500 particles × 128 seeds・同一stable seedで再生し、
  predictive / filtered / post-resamplingの粒子盆地massと系譜を追加計測する。
- 固定exp072予測で`abs(error)>10 ft`が128行以上連続するPF固有の496 wells /
  839 episodesを監査する。
- predictionを変更せず全well parityをgateし、限定12 sentinel wellsだけで
  initialization / transition / GR / resampling / roughening / clampをpaired介入する。

## 検証方針

- Fold: exp072と同じ773-well GroupKFold size balanceをsource row数から再構成
- Group: well
- Stratification: なし。原因比率はfold / sign / tail / episode長別に監査
- Leakage Check: truth / episodeは診断集計だけに使い、PF dynamicsと出力生成には不使用

## 実行入口

- 学習 notebook: `exp410_likpf_particle_resampling_basin_audit_train.ipynb`
- full shard notebook:
  `exp410_likpf_particle_resampling_basin_audit_train_variant0..3.ipynb`
- counterfactual shard notebook:
  `exp410_likpf_particle_resampling_basin_audit_counterfactual_variant0..3.ipynb`
- 推論 notebook: `exp410_likpf_particle_resampling_basin_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp410_likpf_particle_resampling_basin_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 4-well preflightで固定float32予測とのmax abs / RMSE parityがともに0。

### 悪かった点

- full / counterfactual完了前のため原因比率は未確定。

### リスク / 注意

- PF-offset episodeをtruthで固定したtarget-late機構診断であり、candidate CVではない。
- sentinel counterfactualも反証用で、全496 wellsへの改善一般化を主張しない。

## 次

- full 4 shardをstrict mergeし、原因readoutから固定規則でsentinelを作る。
- 12 variantsのpaired counterfactualをKaggle CPUで実行する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
