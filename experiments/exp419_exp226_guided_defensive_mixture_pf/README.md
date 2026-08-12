# exp419_exp226_guided_defensive_mixture_pf

## 状態

- ルート: `pf_beam`
- 状態: train-side完了・technical PASS / scientific FAIL・terminal close
- CV / Public LB / Private LB: `10.680074153 / なし / なし`
- 作成日: 2026-07-27
- scientific PF parent: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- geometry parent: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 原因根拠: `exp410_likpf_particle_resampling_basin_audit`
- 負例境界: `exp281_exp226_residual_offset_exact_hmm_transition_probe`

## 仮説

exp226のabsolute pathやoffsetをPFへ継承するのではなく、fold-safeなgeometry-only
`tvt_geop + Z`の局所rateを粒子配置のproposalにだけ使う。元のPF transitionを50%残す
defensive mixtureと厳密な`p0/q`補正によりtarget posteriorは変えず、exp410で主因だった
finite particle support不足をPF固有のimportance samplingで改善できるか検証する。

## 変更点

通常transitionのrate proposalを50%、exp226 geometry rate中心の
`1x / 4x / 16x`幅Gaussianを各`1/6`とする。position conditional、GR emission、
systematic resampling、roughening、500 particles、128 seedsはexp072と同一である。
seed aggregationはexp404 x1.0のfull-suffix temperature-5 evidence weightingへ固定する。

geometryから使うのは同じouter foldで保存された`well_id / row_idx / suffix_offset /
tvt_geop`だけである。exp226 final、`gr_delta`、U projection、suffix TVT、errorはproposalへ
渡さない。

## HMM案との違い

exp281のような`TVT = exp226 + delta`固定grid再デコードではない。continuous particlesの
proposal sampling、importance correction、ESS resamplingを使い、exp226が外れたwellでも
元transition成分を必ず残す。blend、固定offset state、HMM、adaptive gateは含めない。

## 検証方針

- control: 保存済みexp404 `likpf_scale_5_x1p0`
- candidate: `exp226_guided_defensive_mixture_scale5` 1 variant
- reporting surface: 3,783,989 suffix rows / 773 wells / 5 folds
- mechanism gate: scale5比`>=0.10 ft`、4/5 folds、support外率`>=5 percentage points`
  減、固定episode SSE`>=10%`減、scope / well-tail guardをすべて満たす
- standalone adoption gate: exp226 final `9.427109596582213`比`>=0.03 ft`、
  3/5 foldsを満たす
- mechanismだけ通過した場合は機構支持として記録するが、推論候補へ昇格させない

詳細な式、allowlist、gate、再現性契約は
`docs/legacy/steering/20260727-exp419-exp226-guided-defensive-mixture-pf/design.md`と
`config.yaml`を正とする。

## 実行量

- scientific variant: 1
- candidate PF well-runs: 773
- control PF / exp226 / HMM / Beam rerun: 0
- 128 seeds × 500 particles
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- LightGBM config / trained fold / booster / GPU: `0 / 0 / 0 / 0`
- Kaggle CPU: 4 shards、保守的に各6時間、hard stop各9時間

preflight 1本、full 4 shard、strict merge 1本をKaggle CPUで完了した。
mergeはversion 1、id_no `128974840`である。

## 実装境界

Jupytext percent形式のcompact self-contained train候補、defensive-mixture PF kernel、
truth-late support freeze、4-shard merge、technical / mechanism / adoption gate、
exp419専用testを実装済みである。geometry weight 0 modeはexp404 synthetic fixtureと
bitwise parityを確認した。

正規train Notebookを採用し、preflight / 4 shard / mergeをKaggleで完了した。
inference Notebookは未実装であり、scientific gate FAILのため推論・提出へ進めない。

## リスク

- geometry成分が外れると粒子を浪費し、有限粒子では悪化し得る
- proposalでGR補正版を使うとemissionとの二重利用になるため、`tvt_geop`以外を拒否する
- scale5はfull-suffix GRを使うbatch predictionであり、causal online予測とは呼ばない
- 同じOOFでmixture weight、幅、clip、GR sigma、noise、particle数を救済探索しない

## 所見

technical gateはPASSし、RMSEは保存exp404比`0.234448 ft`、4/5 foldsで改善した。
一方、support外率はexp410比`33.2912 points`悪化し、hidden-like spatial、
by-well p95、worst-well、exp226比較をFAILした。importance correctionでtarget posteriorを
保っても、有限粒子の半数をgeometry proposalへ割くこと自体がsupportを壊すと判断する。

## 次

事前登録どおりsame-OOF rescue、inference、submissionなしでbranchを閉じる。
詳細は`result.md`と`kaggle/output/merge_v1/metrics.json`を正とする。
