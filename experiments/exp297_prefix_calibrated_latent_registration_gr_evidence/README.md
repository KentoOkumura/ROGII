# exp297_prefix_calibrated_latent_registration_gr_evidence

## 状態

- ルート: `pf_beam`
- 状態: Kaggle private CPU version 2完了、`FAIL_STOP_NO_STAGE4`
- CV: H256 expected candidate SSE headroom recovery `-0.116476`
- Public LB: 対象外
- Private LB: 対象外
- Submit ID: 対象外
- 作成日: 2026-07-19
- 親実験: `exp293_physics_only_candidate_bank_headroom_contract`

## 仮説

exp293で固定したdeployable12 candidateを増減せず、known prefixから校正したType Well/horizontal GRの
観測evidenceをlatent registrationとreliable/unreliable状態で周辺化すれば、H256/H512 blockで
低SSE candidateへ十分な確率質量を置ける。

## 変更点

- prefix最大512 rowsのHuber affine calibrationとMAD residual scaleを追加した。
- `candidate_tvt + delta`、`delta=-20..20 ft`の21状態でType Well GRを参照する。
- Student-t residual、NCC、chain-rule derivative残差をblock/state内でrobust標準化する。
- reliable joint posteriorとcandidate/registration marginalをtarget-free artifactとしてSHA freezeする。
- well内finite GRのstable circular shuffleをnegative controlとする。
- freeze後にだけtrue TVTを読み、expected candidate SSE headroom recoveryを集計する。
- inference notebookはraw-test prediction/submissionを常に停止する。

## 検証方針

- Fold: exp293のouter fold 0..4を再利用し、fitは行わない。
- Group: exp293固定non-overlap H128/H256/H512 block。
- Leakage check: raw horizontalはfreeze前に`MD/GR/TVT_input`だけを読み、truth-bearing fileのraw SHAも
  freeze後に計算する。posteriorと入力manifestをclose・再hashしてから専用loaderが`TVT`を読む。
- Primary metric: H256 expected candidate SSE headroom recovery。
- PASS: H256 pooled `>=0.35`、5/5 folds正、realがshuffleをpooled/5 foldsで上回る、H512低下
  `<=0.05`、1000+/hidden-like 2面anchor非悪化、freeze前truth access 0。

## 実行入口

- 実装候補: `exp297_prefix_calibrated_latent_registration_gr_evidence_compact_selfcontained_train.ipynb`
- 停止契約: `exp297_prefix_calibrated_latent_registration_gr_evidence_compact_selfcontained_inference.ipynb`
- tests: `tests/test_exp297_prefix_calibrated_latent_registration_gr_evidence.py`
- canonical train notebookへcompact版をbyte-identicalで採用済み。inferenceは未採用。
- Kaggle train実行は承認済み。inference/submissionはfail closedのまま。

## 結果

| メトリック | 値 |
| --- | --- |
| contract tests | 10 passed |
| H256 anchor / real expected / oracle RMSE | 8.238332 / 8.620041 / 3.552829 |
| H256 real / shuffle recovery | -0.116476 / -0.101397 |
| H512 real recovery | -0.119000 |
| Stage-2 decision | `FAIL_STOP_NO_STAGE4` |
| Public / Private LB | 対象外 |

## 所見

- 3,783,989 rows / 773 wells / 105,818 block-controlを1,070.800秒で完走した。
- truth accessはfreeze前0、freeze後773。取得outputのtarget-free/readout計12 SHAは全一致した。
- H256 real recoveryは5/5 foldsで負、5/5 foldsでshuffleより悪く、1000+とhidden-like 2面もanchorを悪化させた。
- H256 blockのeligible-state coverageは29.5044%、reliable probability中央値は0だった。safe fallbackは機能したが、
  利用可能なreal GR evidenceもcandidate順位を改善できなかった。
- Stage 2はTVT予測を生成していない。固定契約どおりStage 3/4、inference、submissionを閉じる。

## 次

exp297 branchは終了。同一観測posteriorのgrid/weight/prior/threshold救済は行わない。物理routeでは独立設計済みの
exp298 local-shape source監査とexp295 candidate-free SSMだけを、それぞれの既存guardと別承認に従って扱う。
