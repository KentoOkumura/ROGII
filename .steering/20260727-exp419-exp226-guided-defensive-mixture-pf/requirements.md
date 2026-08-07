# 要件

## 依頼

exp226のZ・空間donor主体geometryとexp072 likelihood-PFを、予測値blendや
HMM residual-offset再デコードではなく、PF固有のproposal / importance samplingとして
統合する実験を設計する。

初回承認は`KAGGLE_DIRECTION.md`のbacklog、steering、実験scaffold、設計確定までとし、
PF kernel、Jupytext source、Notebook、test、Kaggle package、実行、推論、提出は
実装しない範囲だった。

2026-07-27の追加依頼`exp419を実装してください`により、train-side PF kernel、
compact self-contained Jupytext候補、target-free support freeze、merge / gate、
contract testまでを追加承認範囲とする。正規Notebook採用、Kaggle package、push、
fixed-probe / full run、推論、提出は引き続き別承認とする。

## 仮説

exp226のfold-safe geometry rateをPFのproposalへ限定し、元transitionを50%残す
defensive mixtureへ`p0/q`補正を適用すれば、HMMのfixed offsetを再現せずに
finite-particle support不足だけを改善できる。

## 制約

- Route: `pf_beam`
- scientific PF parent:
  `exp404_scale5_sigma_gr_likelihood_pf_ablation`の
  `likpf_scale_5_x1p0`
- geometry parent:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- mechanism evidence:
  `exp410_likpf_particle_resampling_basin_audit`
- negative boundary:
  `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- exp226 final prediction、`gr_delta`、U-projected path、truth、errorをPF state、
  emission、proposal、outputへ渡さない。
- exp226のfold-safe `tvt_geop`から計算する局所`TVT+Z` rateだけを、
  PF rate transitionのproposal centerへ使う。
- PFのtarget transitionはexp072のまま固定し、geometry-guided proposalには
  `p/q` importance correctionを必須とする。
- proposalは通常transitionを50%残すdefensive mixtureとし、geometry側は
  exp072 rate noiseの`1x / 4x / 16x`幅を等分した固定3成分とする。
- particles 500、seeds 128、GR sigma x1.0、scale5 seed aggregation、
  initialization、momentum、position noise、ESS threshold、systematic resampling、
  roughening、Type Well grid、GR missing補間を変更しない。
- active scientific variantは1。保存済みcontrolを使い、control PF、
  exp226、HMM、Beam、LightGBMを再実行しない。
- row / well / fold / hidden-like / truthをproposal生成へ使わない。candidate predictionと
  target-free diagnosticsをfreezeした後だけ評価情報を結合する。
- 初回実行はKaggle CPUを正とし、ローカルPF実行は行わない。
- 再現性は`docs/06_reproducibility.md`に従い、per-well stable seed、
  shard-independent RNG、input / code / config / prediction SHAを記録する。
- 正規Notebook採用、package、push、run、inference、submissionはそれぞれ
  train-side実装承認の範囲外とする。

## 受け入れ基準

### Technical

- active scientific variant 1、candidate PF well-runs 773、
  control / exp226 / HMM / Beam rerun 0、LightGBM config / trained fold /
  booster / GPUがすべて0と記録される。
- 3,783,989 rows / 773 wells / reporting folds 0--4を、欠落・重複・fallbackなしで生成する。
- geometry入力がexp226保存OOFの同じouter foldで作られた`geop`だけであり、
  final / GR / truth / error列がproposal freeze前のallowlistに入っていない。
- mixture weightsが`0.5 + 3 * (1/6) = 1`、全importance ratioがfiniteかつ
  `p/q <= 2 + 1e-12`を満たす。
- geometry mixture weightを0にした固定probeで、exp404 x1.0 scale5 kernelとの
  prediction parityをfloat32保存後`1e-6 ft`以内で確認する。
- 保存exp404 scale5 controlのRMSEを`1e-5 ft`以内で再現する。

### Scientific mechanism gate

- candidateがscale5 control RMSE `10.914522073423171`を`0.10 ft`以上改善する。
- scale5 control比で4/5 folds以上を改善する。
- raw-GR observedを`0.10 ft`以上改善し、raw-GR missing、high-missing、
  suffix 1000+、hidden-like spatial / typewell-purgedを各`0.02 ft`より
  悪化させない。
- exp410固定496-well scopeで、truthがmajority-seed predictive particle support外にある
  row率を5 percentage points以上減らす。
- exp410固定episode scopeで、exp404 scale5 control比SSEを10%以上減らす。
- by-well delta RMSE p95を`0.25 ft`以内、worst-well regressionを`2.0 ft`以内にする。

### Standalone adoption gate

- mechanism gate通過に加え、exp226 final OOF RMSE
  `9.427109596582213`を`0.03 ft`以上改善し、3/5 folds以上でexp226を改善する。
- adoption gate FAIL時はPF proposal機構のpositive evidenceとして記録しても、
  inference / submission候補へ昇格させない。
- 全gate FAIL時はmixture weight、proposal幅、importance clip、GR sigma、
  process noise、roughening、seed / particle数、well / row gateを同じOOFで救済探索しない。
- deterministic anchorと呼ぶ場合はinput / code / config / prediction content SHA、
  Kaggle kernel version、rerun parityを記録する。
- gzip生成物はraw gzip SHAとdecompressed content SHAを分け、
  decompressed content SHAを主証拠とする。

## 次のアクション

train-side実装完了で停止する。正規Notebook採用とKaggle package / pushは別承認とし、
実行承認後もfixed-probe technical gateをfull 4-shard runより先に行う。
