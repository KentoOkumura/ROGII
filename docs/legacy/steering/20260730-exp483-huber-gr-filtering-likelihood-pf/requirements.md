# 要件

## 依頼

exp389でexact HMMへ適用したfixed Huber GR emissionを、現行の
temperature-5 likelihood-PFの粒子filtering尤度へ移植する実験を設計する。
backlog、steering、実験ディレクトリを作成し、段階ごとの別承認で実装・実行する。

## 根拠

- exp389はGaussian HMM比でdirect RMSEを`0.085546105 ft`改善し、5/5 foldsで改善した。
- 一方、by-well delta p95は`+0.002234 ft`、worstは`+1.750248 ft`でtail gateをFAILした。
- exp430は凍結済みPF軌道のseed evidenceだけをHuber化した実験であり、粒子重み、
  ESS、resampling、以後の軌道を変えるfiltering尤度は未検証である。

## 制約

- Routeは`pf_beam`。
- 科学的親はexp417、実装親と保存controlはexp404のx1.0 / scale-5 PF。
- 変更は各粒子のGR log emissionだけ。Huber deltaは`1.345`に固定する。
- GR scale、500 particles、128 seeds、PF dynamics、resampling、roughening、
  missing-GR処理、temperature-5 seed集約を変更しない。
- Stage 0はstable-hash fixed32のtechnical preflightだけ。
- Stage 1はStage 0全PASSと別承認がある場合だけ773 wellsを実行する。
- exp404保存controlを使い、control PFは再実行しない。
- 2026-07-30のユーザー依頼でimplementationと専用testを承認済み。
- 正規Notebook採用、Kaggle package、Stage 0は承認・完了済み。
- 2026-07-30のユーザー依頼`Stage1に進んでください`で全773 wellsの
  Stage 1実装、package、push/runを承認済み。
- inference、submissionは未承認。

## 受け入れ基準

- Huber式、固定delta、変更しないPF契約が一意に記載されている。
- exp430との違いが「filter内のparticle likelihood」であると明記されている。
- Stage 0/1の実行量、truth-late、stable seed、SHA契約が固定されている。
- Stage 1の平均、fold、scope、by-well tail、固定HMM/PF blendの全AND gateがある。
- delta、scale、temperature、mixture、gate、particle/seed数のsame-OOF救済を禁止する。
