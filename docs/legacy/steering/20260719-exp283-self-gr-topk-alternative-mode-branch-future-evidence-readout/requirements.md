# 要件

## 依頼

exp282で弱かったself-GR top-1 donor copyを救済するのではなく、same-wellの過去GR一致を
「現在のbase pathとは異なるmode branchの候補提案」に限定し、提案後の未来区間で得られる
typewell尤度とgeometry整合性が正しいbranchを識別できるかを分離して監査する。

初回依頼ではbacklog、実験ディレクトリ、steeringと固定設計だけを作成した。2026-07-19の
追加依頼「exp283の実装を進めてください」により、matching、readout、compact self-contained
train notebook、fail-closed inference、境界testまでを実装対象へ変更する。Kaggle実行、
HMM/PF接続、raw-test推論、提出は引き続き対象外とする。

## 仮説

Assumption: exp282が失敗した主因は、局所GR一致をそのままabsolute TVT donorとしてhard copyした
ことであり、過去matchの候補recall自体が完全にゼロとは限らない。曖昧なmode transition時だけ
baseを保持したままself-GR top-3を代替branchとして提案し、提案窓と重ならない未来256行の
累積typewell尤度で選別すれば、shuffled proposalより正しいmodeを候補化・識別できる可能性がある。

## 制約

- 実験名: `exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout`
- Route: `pf_beam`
- 直接の親: negative readout `exp282_longtail_prediction_zone_self_gr_loop_closure_readout`
- 固定参照:
  - base path: exp263 fixed `exp226_w500_50_50`（OOF RMSE 8.238331）
  - geometry path: group-safe exp226 `tvt_geop`
  - typewell emission: exp209 Gaussian raw-GR emission
  - ambiguity strata: exp236 posterior bimodality、exact-HMM / likPF / exp226 disagreement、
    exp280互換shift-likelihood margin
- 未来evidenceとproposalを行方向で分離する。proposalはevent rowまで、primary verifierはその後の
  256行だけを使用する。128 / 512行は感度readoutのみで、選択・gate調整に使わない。
- proposal bankはsame-wellのvisible known prefixと、eventより256行以上前のprediction zoneに限定する。
- forward / reverse候補を別に生成し、source横断でdeduplicate後、代替branchはglobal top `K=3`に固定する。
- base branchは必ず保持する。候補平均、TVT平均、hard copy、補正predictionは作らない。
- geometryはabsolute modeを単独識別できないため、primary positive evidenceにはしない。有限性、
  `|anchor shift| <= 80 ft`、typewell support、step/rate/curvatureの整合性診断とvetoだけに使う。
- raw true `TVT`、error、oracle、fold別成績はevent、proposal、future evidenceの全tableを凍結し
  content SHAを確定するまで読まない。
- shuffled controlだけstable SHA256 local RNGを使い、real proposal/evidenceはRNGなしとする。
- 0 booster readoutとし、model学習、HMM/PF再生成、raw-test inference、submissionを行わない。

## 固定event契約

- event候補はunknown suffix上で次の4 strataをtarget-freeに作る。
  1. exp236 `bimodal_flag` segmentの終了row。
  2. `|exact_hmm - likpf_mean| >= 10 ft`が128行以上連続したrunの128行目。
  3. `|exact_hmm - exp226_tvt_geop| >= 10 ft`が128行以上連続したrunの128行目。
  4. 直前128行のexp280互換shift score marginがouter-train q20以下のblock終端。
- 同じwellで256行以内に複数候補がある場合は、row昇順で最初だけを残す。stratumは複数flagを保持する。
- trailing 51行のproposal窓と未来256行のverifier窓が存在するeventだけをprimary対象とする。

## 固定proposal / verifier契約

- GR前処理はexp282互換の補間、rolling mean 5、z-normalizeとする。
- causal trailing window長は`17 / 31 / 51`行、primaryは51行、donor strideは3行とする。
- rankはprimary NCC降順、multiscale agreement降順、known-prefix source、forward、donor row昇順の
  deterministic tie-breakとする。25行以内またはanchor差2ft以内の重複候補を除く。
- known-prefix donor anchorはvisible `TVT_input`、prediction-zone donor anchorはexp263 fixed OOFを使う。
- alternative pathはanchorからexp226 `tvt_geop`のrow-to-row incrementを累積して未来へ延長する。
- verifierのprimary scoreは未来256行のexp209互換mean cumulative raw-GR/typewell log-likelihoodとする。
- geometry veto後、base + top-3のうちtypewell score最大をreadout上のselected branchとする。
- evidenceはproposal NCCを再利用しない。ただしbase/exp226自体がraw-GR派生を含み得るため、
  「統計的に完全独立」ではなく「proposal窓・score関数から分離した検証」として表記する。

## 受け入れ基準

- technical guard: canonical 3,783,989 rows / 773 wells / 5 folds、event/proposal/evidence finite coverage 1.0、
  forbidden truth列0、truth-before-freeze 0、source/branch identity一意。
- proposal guard:
  - post-freeze top-3 `within10` lift vs same-well shuffled donorがpooled `>= +0.02`。
  - 同liftが5/5 foldsで正。
- verifier guard:
  - branch-vs-baseの未来256行RMSE改善labelに対するtypewell score-margin AUCが各fold `>= 0.60`。
  - selected branchのtriggered-block RMSEがbaseよりpooled `>= 0.10 ft`改善し、5/5 foldsで悪化しない。
  - baseがpost-freeze unique-bestのeventでfalse switch率が`<= 0.05`。
- candidate recall、unique-best率、AUC、MRR、score margin、selected RMSE、false switch、
  128/256/512 horizon、4 strata、known/prediction donor、forward/reverse、1000+、hidden-like、
  by-well、shuffled controlを記録する。
- 全scientific guard PASSだけがexp284実装を許可する。FAIL時はK/window/stride/horizon/margin/vetoの
  rescue grid、decoder接続、inference、submitへ進まない。

## 初回設計マイルストーン

- steering 3文書、実験ディレクトリ、planned config / docs / metricsが未記入項目なしで設計確定状態を示す。
- `K=3`、primary horizon 256、event、proposal、evidence、freeze順序、guard、禁止事項が固定されている。
- notebookはtemplate stubのままで、実装・Kaggle package・実行を行わない。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`へ設計済み未実装として登録する。

## 追加実装マイルストーン

- 正規stub notebookは上書きせず、compact self-contained Jupytext train/inferenceを別名で作る。
- event、proposal、evidence、post-freeze truthの4境界を実装し、合成testでfail-closedを確認する。
- 128/256/512 horizon、real/shuffled proposal、fold/scope/by-well readoutと全guardを実装する。
- Jupytext round-trip、`py_compile`、ruff、targeted pytest、strict `validate-exp`を通す。
- 実装後も`execution.kaggle_push_approved=false`を維持し、CPU実行は実行量を再提示して別承認を得る。
