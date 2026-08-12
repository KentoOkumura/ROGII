# 設計

## アプローチ

exp373のdesign-only構成とexp371のfixed13 selector実装を参照し、corrected exp264
fixed12へ`prefix_rate_exact_hmm`を1本だけ追加する。exp362 OOFは
`well_id,row_idx`でcanonical keyへ揃え、source foldをprovenance-onlyとして
exp263 selector foldへrepartitionする。

候補追加後にStage Aで13候補用candidate-long schemaを再freezeし、corrected exp264と
同じ`pred_abs_error` / `p_within10`のnested Stage C selectorだけを40 CPU boosters
学習する。別のnovelty監査は置かず、oracle/add-one headroomはprediction freeze後の
診断出力として同じrunに含め、学習・gate・閾値選択には使わない。

exp362のlocal donor gradientは採用0 segmentだったため、候補の意味は
「known prefixのrateを遷移平均に保つresidual exact HMM」と固定する。donor ledger、
local gradient、support/fallback、`mu_rate`は入力・特徴・候補名から除外する。

## 実験範囲

- 対象実験: `exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264`
- Route: `ensemble`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp362_segment_local_donor_slope_exact_hmm`
- 構成参照:
  - design-only: `exp373_exp355_fixed13_dual_selector_on_exp264`
  - fixed13実装: `exp371_exp333_fixed13_dual_selector_on_exp264`
- 変更する変数: candidate inventoryへ`prefix_rate_exact_hmm`を1本追加する。
- 固定する変数: fixed12値・順序、fixed fallback 7本、既存formula、
  outer/inner fold、objective、sampling、LightGBM config、seed、
  raw-test-safe context、保存済みparent score。
- primary domain: 既存11候補 + `prefix_rate_exact_hmm`。
- fixed fallback domain: 既存7候補のまま。
- native confidence:
  - `candidate_std -> sigma_tvt`
  - `hmm_loglik -> source_loglik`
  - `hmm_loglik / evaluation_row_count -> loglik_per_row`
  - 上記がfiniteの行だけ`confidence_valid=true`
- compact meta: 親74列から77列を予定。追加は2 objectiveの候補scoreと
  primary top1 one-hotであり、Stage A feature数はaudit結果でfreezeする。
- 学習量: 1 variant × 2 objectives × 5 outer × 4 inner = 40 CPU boosters。
- parent/control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / false / false / false`。
- 実装状態: 2026-07-24の追加ユーザー指示によりimplementation-onlyまで完了。
  compact self-contained trainとfail-closed inference候補、helper、contract、
  testを実装済み。正規notebook採用とKaggle runは未承認。

## 入力契約

- exp362 OOF:
  `exp362_segment_local_donor_slope_exact_hmm_oof_predictions.csv.gz`
- prediction logical SHA:
  `bdf616e00bdebb496093d3d05526aebce01381281c4b1c46f7b77e72e57415cb`
- prediction decompressed SHA:
  `e1d672ff9743b92c33a40bec8d4cf3b0a8c29cdbbb37948992f0809522e3e7ef`
- allowlist:
  `well_id,row_idx,fold,candidate_tvt,candidate_std,hmm_loglik`
- source fold role:
  exp362 OOF provenance only。selector feature、eligibility、model inputには使わない。
- join:
  global `well_id,row_idx`完全一致後にexp263 selector foldへrepartitionする。
- forbidden:
  truth/error/oracle、local-gradient `mu_rate`、donor support/fallback、segment ID、
  exp362の評価後readout。

## 比較と判定

- 一次比較: saved corrected exp264 fixed12 hard selector
  `8.652531955610227`。
- 参考比較: fixed fallback `exp226_w500_50_50`
  `8.238331546485645`。これはprimary hard selectorの置換gateにはしない。
- selector score:
  `pred_abs_error`と`p_within10`をouter-train candidate priorと比較する。
- usage:
  `prefix_rate_exact_hmm`のprimary top1率をpooled/fold/distance/hidden-likeで記録する。
- performance:
  pooled、5 folds、near 0--250、1000+、hidden-like spatial、
  hidden-like typewell-purgedをsaved fixed12とrow-paired比較する。
- safety:
  improved/worsened wells、by-well delta median/p95、worst wellを記録する。
- diagnostic-only:
  fixed12対fixed13のH512/whole-well oracle headroomとstrict unique-best率は、
  selector予測freeze後にだけ計算し、同一OOF rescueには使わない。
- decision:
  pooled改善だけではdownstreamへ進めない。事前固定したfold/scope/tail gateを
  すべて満たした場合にだけ、別承認でdownstream TVTを検討する。

## 再現性設計

- seed policy: seed 42。sampling seedはstage/fold/objectiveのimmutable keyから
  SHA256で生成する。
- stochastic 処理の有無: selector samplingとLightGBM CPU training。
  exp362候補は保存済みOOFを読み、HMMやdonor fieldを再生成しない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済みfixed12と
  exp362 OOFだけを使う。
- 並列処理と乱数の関係: samplingは固定key seed、LightGBMは
  `deterministic=true`、`force_col_wise=true`、`n_jobs=8`。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU/internet off。
  exp371実績から約2時間を想定する。
- train cache / test feature regeneration の SHA 記録方針:
  exp263 manifest/catalog SHA、exp362 prediction logical/decompressed SHA、
  parent exp264 score SHA、feature schema/content SHAを記録する。
- model manifest / prediction / submission SHA 記録方針:
  40 selector model SHA、25 compact partition SHA、outer-valid candidate score SHA、
  hard selector OOF SHAを記録する。inference/submission SHAは今回対象外。
- Kaggle package bootstrap 確認方針: prepare後にembedded `config.yaml`、helper、
  kernel sources、CPU/internet/run-on-push、notebook/support ZIP SHAを照合する。
- deterministic anchor: exp362 source候補のrerun parityが未確認のため、
  selector runが再現しても現時点では`false`を維持する。

## リスク

- リークリスク: exp362 OOFと後段評価truthを同時に開くと、oracle/error列が
  feature freezeへ混入し得る。target-free allowlist loader、truth-access counter、
  freeze後のlate truth attachで禁止する。
- semantic risk: exp362という実験名をそのまま候補名にすると、採用0だった
  donor-slopeを評価しているように誤読する。候補IDと説明を
  `prefix_rate_exact_hmm`へ固定する。
- foldリスク: exp362 source foldとexp263 selector foldは異なる可能性がある。
  source foldはprovenance-onlyとし、global key join後のselector-fold
  repartitionと5×5 overlapをmanifest化する。
- CV/LB不一致リスク: exp362はexp209比でpooled `-0.776610 ft`、hidden-likeも
  改善した一方、改善foldは3/5、worst wellは`+52.741426 ft`だった。
  exp371でもselector平均改善とtail悪化が分離したため、自動推論化しない。
- raw-test parity risk: exp362のcurrent-test候補は未生成。selector-only trainが
  通っても、raw-test HMM再生成とnative confidence parityが成立するまで
  downstream/inferenceへ進めない。
- ランタイム/メモリリスク: 13候補long tableは約49.2M outer-valid行。
  exp371と同じchunk/row capを維持する。
- 再現性リスク: exp362はdeterministic anchorではなく、gzip raw SHAもmetadataで
  変わり得る。logical/decompressed SHAをhard evidenceにする。
