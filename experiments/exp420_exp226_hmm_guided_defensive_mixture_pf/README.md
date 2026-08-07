# exp420_exp226_hmm_guided_defensive_mixture_pf

## 状態

- ルート: `pf_beam`
- 状態: 実装完了・exp411 schedule prerequisite FAIL・未実行
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-27
- 実装lineage parent: `exp419_exp226_guided_defensive_mixture_pf`
- scientific PF parent: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- geometry parent: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- HMM kernel parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- HMM schedule reference: `exp411_predictive_filtered_rate_innovation_destick`

## 仮説

exp226のfold-safe geometry局所rateで空間的なparticle supportを増やし、untreated HMMの
predictive-to-filtered rate innovationが発火した32 transitionsだけtarget well自身の
rate変化方向へparticleを追加する。元PF transitionを50%残し`p0/q`補正を行えば、
exp226 / HMMのabsolute datum driftを継承せず、PFの有限粒子support不足と
128 seed算術平均multiplicityを同時に緩和できる。

## 変更点

- inactive:
  元transition `0.5` + exp226 geometry中心`1x / 4x / 16x`各`1/6`
- active:
  元transition `0.5` + geometry 3成分各`1/12` +
  `mu0 + direction*0.005`中心のHMM 3成分各`1/12`
- rate importance correction:
  `p0/q`、clipなし、構成上`<=2`
- HMM schedule:
  CUSUM drift `0.01`、threshold `1.0 rate cell`、activation `32`、
  refractory `128`
- seed aggregation:
  exp404 temperature-5 full-suffix log-evidence weighting
- 最終出力:
  `exp226_hmm_guided_defensive_mixture_scale5` 1 variantのみ

HMM posterior mean / backward message、exp226 final / GR correction / U projection、
prediction blend、selector、ML modelは使わない。

## 検証方針

- Stage 0:
  exp411 fixed32とexp410 fixed12 sentinelの重複なしunion 44 wells。
  direction / lead / control activation、PF support / episode SSEだけをmechanism
  preflightし、selection-biased pooled RMSEはpromotionに使わない。
- Full:
  3,783,989 suffix rows / 773 wells / 5 reporting folds、candidate 1 variant。
- Group:
  `well_id`。
- Leakage:
  HMM schedule、candidate、target-free diagnostics、SHAをfreezeしてからtruth /
  error / role / cause / fold / hidden-likeをlate joinする。
- control:
  保存済みexp404 scale5。親PF / HMM / exp226は再実行しない。
- mechanism gate:
  scale5比`>=0.10 ft`、4/5 folds、support外率`>=5 points`減、
  exp410 episode SSE`>=10%`減、exp408 episode SSE`>=5%`減、
  scope / well-tail guardをAND判定する。
- standalone gate:
  exp226 final比`>=0.03 ft`、3/5 folds。
- physical-anchor gate:
  exp263 fixed physical blend比`>=0.03 ft`、3/5 folds。

## 実行入口

- 学習 notebook: `exp420_exp226_hmm_guided_defensive_mixture_pf_train.ipynb`
- 推論 notebook: `exp420_exp226_hmm_guided_defensive_mixture_pf_inference.ipynb`
- compact候補:
  `exp420_exp226_hmm_guided_defensive_mixture_pf_compact_selfcontained_train.ipynb`
- compact候補はuntreated HMM schedule、scheduled PF、fixed44/full orchestration、
  truth-late readout、fail-close gateをself-containedで実装済み。
- 正規train / inference Notebookはtemplate placeholderであり、実行不可。
- 正規Notebook採用は別承認とする。
- Kaggle Notebook実行を正とし、ローカル実行は明示依頼されたsmoke debugだけとする。

## 設計上の実行量

- active scientific variant: 1
- Stage 0: HMM signal / candidate PF `44 / 44` well-runs、
  5,632 seed-well trajectories、2,816,000 particle starts
- Full: HMM signal / candidate PF `773 / 773` well-runs、
  98,944 seed-well trajectories、49,472,000 particle starts
- control HMM / PF / exp226 rerun: 0
- LightGBM config / trained fold / booster / model / GPU:
  `0 / 0 / 0 / 0 / 0`
- Full Kaggle CPU: 4 shards、保守7.5時間 / hard stop 9時間 per shard

これは実装済みconfigの固定値であり、Kaggle実行の承認ではない。

## 所見

### 良かった点

- 3モデルのabsolute predictionを平均せず、各モデルの有用なrate-level情報だけを
  PF proposalへ統合するため、共通のdatum driftを直接継承しない。
- 元transition 50%と`p0/q`補正により、proposalを増やしても無限粒子極限の
  target posteriorを変更しない。

### 悪かった点

- HMM方向やgeometry rateが外れると有限粒子を浪費するため、理論上のimportance
  correctionだけでは有限粒子での改善を保証しない。
- HMM forwardとPFの両方が必要で高コストである。
- 2026-07-28に同じCUSUM schedule / fixed32を使うexp411 Stage 0が、方向一致
  `0.225397`、passing folds `0 / 5`、control active-row fraction `0.136119`、
  persistent-control active-well差`0.0`でmechanism FAILした。exp420の事前固定
  direction / control gateも同じため、現行契約のStage 0は前提を満たさない。

### リスク / 注意

- Stage 0 fixed44は原因enriched sampleなのでpooled CVとして解釈しない。
- full-suffix evidence weightingはbatch / non-causalである。
- 同じOOFを見たthreshold、proposal weight / width、noise、particle / seed数の
  救済探索は禁止する。

## 次

compact self-contained train候補とcontract testsは実装参照として保持する。
現行契約の正規Notebook採用、package、Kaggle Stage 0 fixed44 runへは進まない。
再開する場合はexp411 scheduleをそのまま使わない独立仮説として再設計し、別実験・
別承認にする。full、inference、submissionは未実行。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
