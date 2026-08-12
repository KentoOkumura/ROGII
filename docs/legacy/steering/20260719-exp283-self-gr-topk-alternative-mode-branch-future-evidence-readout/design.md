# 設計

## 結論

exp282のdonor copyを調整せず、`event detection -> causal self-GR top-3 proposal -> future evidence
freeze -> truth readout`を分離した0-booster branch-and-verify監査とする。primary horizonは256行、
代替branch数は3、base branchは常時保持する。

## 実験範囲

- 対象: `exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout`
- Route: `pf_beam`
- 親: `exp282_longtail_prediction_zone_self_gr_loop_closure_readout`
- comparison anchors: exp263 fixed 8.238331、exp209 HMM/likPF best 10.269696
- 変更する変数: self-GR一致の用途をabsolute donor-transferからtop-K mode proposalへ変え、
  proposal後の未来typewell evidenceでbranch separabilityを測る。
- 固定する変数: K=3、H=256、exp282 GR前処理、exp209 emission、exp226 geometry increment、
  event trigger、tie-break、shuffle、truth attachment順序、guard。
- 対象外: prediction correction、decoder commit、HMM/PF再実行、model fit、inference、submission。

## Stage 0: target-free event table

raw horizontal identity、exp236 safe posterior flags、exp209 `exact_hmm` / `likpf_mean`、exp226
`tvt_geop`だけからeventを作る。exp236 row summaryは`target_tvt_readout_only`を含むため、readerは
safe allowlist以外をdropするのではなく、forbidden列を検出した入力frameをscore APIへ渡せない形にする。

4 triggerはrequirementsの固定条件で作り、同well 256行refractoryを適用する。foldはthresholdの
outer-train q20計算と後段stratificationにだけ使い、truth/errorは読まない。event tableを保存し、
schema/content SHAを固定する。

## Stage 1: causal self-GR proposal

event rowを`e`とし、receiverは`[e-50, e]`のcausal trailing GRとする。donorは次のunionである。

1. visible known prefixで51行windowが収まるcenter。
2. prediction zoneでwindow終端が`e-256`以前のcenter。

GRはwell内linear interpolation、finite fallback、rolling mean 5、window z-normalizeを行う。
17/31/51行のforward/reverse NCCを計算し、51行NCCをprimaryにする。source/orientationをまたいで
lexicographic rankし、donor center 25行以内またはanchor TVT 2ft以内を同一proposalとしてdeduplicate、
global top-3を残す。bank quotaは設けない。source別recallは別集計する。

anchorはknown donorならvisible `TVT_input`、prediction donorならfrozen exp263 fixed prediction。
event以降のbranchは次で作る。

`branch_tvt[r] = anchor_tvt + tvt_geop[r] - tvt_geop[e]`

orientationはproposal featureとして残すが、future pathのstep signを反転しない。反転するとgeometryと
同時に変更する別仮説になるためである。base branchはfrozen exp263 fixed pathそのものとする。

stable shuffled controlはreal proposal freeze後、same well / event / source bank内でdonor assignmentだけを
置換する。seedは`SHA256(exp_name, 42, well_id, event_row, source, "donor_shuffle")`から作る。

## Stage 2: future evidence

proposalが使用したrowより後だけをscoreする。primary windowは`e+1 ... e+256`、diagnosticは128と
512（存在するeventのみ）である。

- typewell evidence: exp209互換Gaussian raw-GR log-likelihoodの累積mean。proposal NCC、gap、
  multiscale scoreは加えない。
- geometry diagnostics: anchor shift、donor/receiver local rate差、step、curvature、typewell support外率。
- hard veto: non-finite、`|anchor shift| > 80 ft`、typewell support + 40ft外、step非finite。
- selection readout: veto後のbase + top-3でH256 typewell evidence最大。全alternativeがvetoならbase。

geometryはconstant-offset branchのabsolute正解を決められないため、combined weighted scoreは作らない。
未来typewell evidenceとgeometry tableをtruthなしで保存し、event/proposal/evidenceのcontent SHAを
manifestへ固定してからだけStage 3へ進む。

## Stage 3: post-freeze truth readout

raw true TVT、5 fold、distance、hidden-like roleをidentity joinする。

- proposal: event anchor within2/5/10、top-1/top-3 recall、MRR、unique-best、oracle H256 RMSE headroom。
- verifier: branch-pair AUC、truth-best MRR、selected branch、score margin、selected/base/shuffled RMSE。
- safety: base unique-best時false switch、最大well回帰、fold、4 trigger、source、orientation、
  1000+、hidden-like、H128/H256/H512。

branch-pair AUCのlabelは`alternative H256 RMSE < base H256 RMSE`、scoreは
`LL_alternative - LL_base`とする。同点はnegative、単一class foldはguard FAILとする。

## 固定guardと判断

- Technical: identity / finite / coverage / freeze orderを全PASS。
- Proposal: top-3 within10 lift vs shuffled `>=0.02`、5/5 folds positive。
- Verifier: AUC `>=0.60` in 5/5、selected H256 RMSE gain `>=0.10 ft`、5/5 fold非悪化、
  base unique-best false switch `<=5%`。

PASSしても補正候補にはしない。exp284のmasked recovery backtest実装へ進む許可だけを与える。

## 実行契約

- active readout variants: 1
- model / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
- HMM / PF regeneration: 0 / 0
- parent/control retraining: 0
- runtime: Kaggle CPU、GPU/internet off、single process、well逐次、event chunk
- inference/submission: disabled / disabled
- Kaggle push approval: false。実装後に実行量を再提示して別承認を得る。

## 再現性設計

- real event/proposal/evidenceはRNGなし。shuffleのみstable SHA256 per-event local RNG。
- sorted well / event / donor orderとdeterministic tie-breakを固定する。
- exp236 row summary raw/decompressed SHAは
  `5410ceae...30a9` / `bf124fda...e9a0`。safe列だけを専用readerで読む。
- gzipはrawとdecompressed SHAを分け、decompressed contentを主証拠にする。
- event/proposal/evidenceそれぞれのschema、logical content、file SHAとscientific contract SHAを保存する。
- model / prediction / submissionを作らないため、それらのSHAは対象外。
- 初回成功だけでdeterministic anchorとは呼ばず、fixed-input diagnosticとする。
- Kaggle prepare時はloose config/sourceとbootstrap manifest、CPU/offline metadataを照合する。

## 予定生成物

- `exp283_*_event_contract.json`
- `exp283_*_target_free_events.csv.gz`
- `exp283_*_target_free_proposals.csv.gz`
- `exp283_*_target_free_future_evidence.csv.gz`
- `exp283_*_proposal_metrics.csv`
- `exp283_*_verifier_metrics.csv`
- `exp283_*_fold_metrics.csv`
- `exp283_*_scope_metrics.csv`
- `exp283_*_by_well_metrics.csv`
- `exp283_*_input_manifest.csv`
- `exp283_*_summary.json`

実装時は`*`を完全なexperiment nameへ展開する。

## 実装反映（2026-07-19）

- `*_compact_selfcontained_train.py` / `.ipynb`を別名で実装し、ユーザーの実行承認後に正規notebookへ採用した。
- train sourceは10章・23 cells相当で、safe reader、4 event strata、outer-train q20、causal
  forward/reverse top-3、stable shuffled donor、H128/256/512 evidence、geometry veto、post-freeze
  truth、proposal/verifier/fold/scope/by-well guardまでself-containedに展開する。
- exp209 enriched cacheは実ファイルに`likpf_mean`がないため、safe列
  `hmm_mean_tvt - hmm_minus_likpf_mean`で同値復元し、manifestへderivationを記録する。
- multiscale agreementは固定lexicographic tie-break専用として`mean(NCC17, NCC31)`を使い、
  primary順位は常にNCC51を先にする。
- inference notebookはfail-closedで、decoder、test prediction、submissionを生成しない。
- Kaggle CPU version 2を完走し、technical guardは全PASS、scientific guardはFAILした。proposal liftと
  5-fold AUCはPASSしたが、selected gain `-6.384973 ft`、nonregressing fold 0/5、false switch
  `55.5647%`のため、設計どおり救済grid・decoder接続・inference・submitを閉鎖した。

## リスク

- exp282同様、GR motifは異なるTVTで反復する。top-K recallとshuffled liftを先にguardする。
- future typewell evidenceもraw GR由来で、proposalと統計的に完全独立ではない。row非重複、score非共有、
  freeze境界を守り「独立」と過大表記しない。
- exp236 posteriorはfull-well smoothing由来であるため、triggerはseparability stratumであり
  online-causal detectorの証明にはしない。
- geometryはoffset modeを決められない。positive evidenceへ重み付けせずveto/diagnosticに限定する。
- eventをtruth/errorで選ぶとoracle化する。target-free trigger以外は禁止する。
- O(n^2)を避けるためdonor stride、256行gap、well/event逐次、full pair matrix非保存を固定する。
