# exp495_uncertainty_weighted_exp226_rate_observation_hmm セッションノート

## 目的

exp209 exact HMMのrate追従遅れに対し、exp226 geometry rateをhard置換せず、
known-prefix由来の不確実性を持つGaussian rate観測としてsoftに融合する設計を固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage_0b_completed_fail_closed`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV / LB: なし
- Stage 0A実装 / 正規Notebook採用 / Kaggle package: 完了
- Stage 0A run: Kaggle private CPU version 1で完了
- Stage 0B: ユーザー明示overrideでKaggle private CPU version 4を完了、FAIL
- Stage 1 / inference / submission: 未実施・閉鎖

## 2026-07-31 設計記録

ユーザー指示:

```text
バックログ、実験ディレクトリ、steeringを作成して設計を確定させてください。
実装はまだです。
```

`kaggle-review-exp`に従いsteeringを先に作成し、その後exp495 scaffoldを作成した。
`kaggle-strategy`に従い、Late phaseの既存anchorとrate/HMM失敗系列を確認した。

根拠:

- exp408: persistent offsetの主要因はcurrent GRではなくforward rate-prior hysteresis。
- exp355: exp226 geometry相対rate scheduleでexp209 `11.938287235`から
  `11.291976616 ft`へ改善し5/5 folds改善したが、hidden-like 2面とworst
  `+52.743754 ft`をFAILした。
- exp411: HMM内部CUSUM triggerはfuture方向一致`0.225397`、0/5 foldsでFAIL。
- exp491: exp226 final差分のhard TVT-only遷移はfixed32で`+4.314194 ft`悪化。
- exp285: known-prefix offsetはfull-suffix offsetを予測できず、prefix transferは未確認。

このため、exp355のrate centerは維持しつつ、fixed scheduleではなく
known-prefix tail128のrobust rate残差scaleをGaussian観測分散に使う独立仮説とした。
prefix bias補正は行わず、信頼度だけを推定する。

## 科学式

```text
r_geom[t] = Δ(tvt_geop[t] + Z[t]) / ΔMD[t]
mu_226[t] = r_prefix_exp209 + r_geom[t] - r_geom[first_segment]
sigma_226[w] = max(0.002, 1.4826 * MAD(last128 prefix rate residual))
P495_t(j|i) ∝ P209(j|i) * exp(-0.5*((r_j-mu_226[t])/sigma_226[w])^2)
ΔTVT = r_j * ΔMD - ΔZ
```

valid prefix transitionが32未満なら観測factorをuniformにし、exp209へfallbackする。
exp226 final`tvt_pred`、GR correction、U projectionは使用しない。

## 段階と実行量契約

| 段階 | diagnostic/scientific variant | HMM well-runs | control再実行 | model | booster | PF/Beam | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0A | diagnostic 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stage 0B | scientific 1 | 32 | 0 | 0 | 0 | 0 | 0 |
| Stage 1 | scientific 1 | 773 | 0 | 0 | 0 | 0 | 0 |

2026-07-31の追加ユーザー指示`exp495を実装してください`により、Stage 0Aの
実装だけが承認された。実際のrunはすべて0。
Stage 0A / 0B / 1はそれぞれ全gate PASS、結果レビュー、別承認を必要とする。

## 2026-07-31 Stage 0A実装記録

実装したもの:

- `exp495_*_compact_selfcontained_train.py`と候補`.ipynb`。
- exp226 OOFを`well_id,row_idx,suffix_offset,fold,tvt_geop`だけで読むstrict allowlist。
- exp226 saved fold kappaとouter-fold donor fieldを使うgeometry-only replay。
- target wellの最後の128個のfinite observed U-rate / positive `ΔMD` transitionを選び、
  最初の選択transition直前で`TVT_input`をmaskしてprefix geometryを再生する処理。
- `sigma_226=max(0.002,1.4826*MAD(residual-median(residual)))`、valid 32未満の
  observation無効化、exp355互換`mu_226` suffix schedule。
- prefix transition / uncertainty / suffix schedule / segment ledgerをtruth join前に
  logical SHA freezeし、変更後のlate joinを拒否するguard。
- pooled / fold Spearman、low/high sigma half rate RMSE、low-sigma exp355 schedule gain、
  fallback率を固定ANDで判定するStage 0A gate。
- 18件の契約テスト。U-rateは`Δ(TVT+Z)/ΔMD`であり、`ΔZ`を二重加算しない。

親compactとの比較:

- exp355 Stage 0 compactは8章 / 1,361行。
- exp495 Stage 0A compactは8章 / 2,194行（exp226 prefix replay本体をself-containedに
  含むため増加）。config表示、入力guard、replay、freeze、truth late-join、metric、
  gate、生成物保存をNotebook上で追える。
- `__file__`依存は0。同一exp helper importも0。

## 戦略上の位置づけ

現在はLate phaseで、ML Public-LB anchorはexp413のCV `7.884802794` / Public LB
`7.201`、ensemble anchorはexp082 Public LB `7.601`。direct HMM exp209は
CV `11.938287235`で、物理routeのCV/LB順位は安定していない。
exp495は最終提出のP1/P2を追い越さないP3の高リスクCPU mechanism実験とする。
手堅い部分はStage 0Aの0-HMM falsificationであり、当たれば大きい部分だけを
Stage 0B exact HMMへ進める。

## コマンドログ

実行済み:

```bash
make new-steering EXP=exp495_uncertainty_weighted_exp226_rate_observation_hmm
make new-exp EXP=exp495_uncertainty_weighted_exp226_rate_observation_hmm
.venv/bin/python -m pytest -q experiments/exp495_uncertainty_weighted_exp226_rate_observation_hmm/tests/test_exp495_contract.py
.venv/bin/ruff check experiments/exp495_uncertainty_weighted_exp226_rate_observation_hmm/exp495_uncertainty_weighted_exp226_rate_observation_hmm_compact_selfcontained_train.py experiments/exp495_uncertainty_weighted_exp226_rate_observation_hmm/tests/test_exp495_contract.py --select F821,F401,F841,E9
.venv/bin/python -m py_compile experiments/exp495_uncertainty_weighted_exp226_rate_observation_hmm/exp495_uncertainty_weighted_exp226_rate_observation_hmm_compact_selfcontained_train.py experiments/exp495_uncertainty_weighted_exp226_rate_observation_hmm/tests/test_exp495_contract.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp495_uncertainty_weighted_exp226_rate_observation_hmm/exp495_uncertainty_weighted_exp226_rate_observation_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp495_uncertainty_weighted_exp226_rate_observation_hmm/exp495_uncertainty_weighted_exp226_rate_observation_hmm_compact_selfcontained_train.py
```

結果: pytest `18 passed`、ruff `All checks passed`、py_compile / Jupytext test PASS。
`task validate-exp ...`は環境に`task`コマンドがなく実行前に終了したため、同等の
`make validate-exp EXP=exp495_uncertainty_weighted_exp226_rate_observation_hmm`へ切り替え、
strict validation PASSを確認した。

全体回帰確認:

```bash
make test
```

`1783 passed / 8 skipped / 4 failed`。失敗は今回未変更の既存領域だけだった。

- exp293: `downstream_branch_contract.md`の実ファイルSHAと固定期待SHA不一致 2件。
- exp296: 完了済みconfigのstatus / `run_variant`と旧テスト期待の不一致 2件。

exp495固有18件、notebook静的検査、strict experiment validationはすべてPASSしている。

## 2026-07-31 Stage 0A実行承認とpush前記録

ユーザー指示`実行してください`を、正規train Notebook採用、Kaggle package作成、
canonical private CPU kernelへのStage 0A push/run承認として記録した。

実行量:

| 項目 | 数 |
| --- | ---: |
| diagnostic variant | 1 |
| 対象well | 773 |
| reporting fold | 5 |
| HMM well-run | 0 |
| fitted ML model / LightGBM config / trained fold / booster | 0 / 0 / 0 / 0 |
| PF / Beam / GPU | 0 / 0 / 0 |
| 親control再実行 | 0 |

package:

- kernel id: `kentookumura/exp495-uncertainty-weighted-rate-obs-hmm-train`
- title: `exp495 uncertainty weighted rate obs hmm train`
- 初回の長い title は Kaggle の50文字上限で `400 INVALID_ARGUMENT` となったため、
  仮説や実行内容を変えずに46文字へ短縮した。
- private / CPU / internet off / run-on-push
- kernel source: exp226 train、exp209 train
- 実行時loose / embedded config SHA: `6bb66d246aa0c6ee480f0a1853a7a51fc38491d48207a7ebe1404b21b4b33b0e`
- 実行時正規Notebook SHA: `0210b394de908e8c0ea31a21d75cf8b8fd4aaecec40c803fc1392bb50de7b088`
- 実行時bootstrap済みpackage Notebook SHA: `5265b914362ecf0e0bde624a3304d1674b7ac8e87ce1a91b99605ef245d22ded`

Stage 0Bへは結果にかかわらず自動昇格しない。

未実行:

```text
ローカルNotebook実行、Stage 0B / Stage 1 HMM、inference、submission
```

## 2026-07-31 Stage 0A Kaggle実行結果

- kernel: `kentookumura/exp495-uncertainty-weighted-rate-obs-hmm-train`
- version / id_no: `1` / `129285050`
- private / CPU / internet off、status `COMPLETE`
- runtime: `179.778713361 sec`
- Stage 0A: 1 diagnostic、773 wells、3,783,989 suffix rows、5 folds
- HMM / model / booster / PF / Beam / GPU / control再実行: すべて0

technical gateは全件PASSした。missing / duplicate / forbidden exp226 columns /
truth read before freezeは0、formula parity max absは0.0、uncertainty coverageは1.0。

mechanism gateはFAILした。

| check | 値 | 閾値 | 判定 |
| --- | ---: | ---: | --- |
| pooled sigma vs suffix abs rate error Spearman | 0.088435265 | 0.20以上 | FAIL |
| positive Spearman folds | 5 | 4以上 | PASS |
| low/high sigma rate RMSE gain | 0.076454421 | 0.10以上 | FAIL |
| low-sigma exp355 schedule gain | 0.088438180 | 0.05以上 | PASS |
| improving schedule folds | 4 | 4以上 | PASS |
| fallback well fraction | 0.0 | 0.05以下 | PASS |

low/high sigmaのRMSEは`0.012081677 / 0.013081841`、fallbackは0/773だった。
prefix MADは方向性を5/5 foldsで持つ一方、pooled順位相関と群間分離が弱く、
unknown suffixのgeometry-rate信頼度を十分に識別するという前提は不成立と判断する。
decisionは
`close_before_hmm_implementation_without_window_sigma_scale_or_gate_rescue`。

主要logical SHA:

- input manifest: `892e9176813dca840dab52e342ed64cfbce8936c27d1f6785b6d24b7c3eff272`
- uncertainty: `fb6411fac56efb5913ecdd7cc7a418bc64f2d352bf2f69a04100177b32e94c3b`
- suffix schedule: `f56edf7f11875a119a989145b85a3ce55b8ec8c85f68e4c4e9d45428fbdb9bdc`
- suffix rate readout: `65ffdc9fb6c112a807e61c9bfca36af3a7ac20a2fca8554a5c898e3f0a86fd5e`
- scientific contract: `d951375f63a309a74b8596fe8d9da754bae29874c5ee5e87fa4fbd85c619af8d`

再実行を防ぐため、結果記録後のlocal configは`run_stage_0a: false`へ戻し、
statusを`stage_0a_completed_fail_closed_before_hmm`へ更新した。post-run config SHAは
`0db8c73bf7555b9e7793f1d4b0aca20772e9d733b621feac0b5d33beb275ad65`、
post-run正規Notebook SHAは
`cc7d8c43e3911345601369c455e9b406a8e4fc6d6b1136275390c4ce0596cd12`。

Stage 0Bは自動昇格せず、契約どおり実装・実行しない。同一OOFでprefix window、
sigma floor/scale、temperature、threshold、gateを調整せず、PF/selector/blend救済も
行わない。保存済みartifactだけを使うprefix-to-suffix regime-shift原因分解を
低優先度の次案として全体backlogへ追加する。

## 再現性メモ

- seed policy: RNGなし、fold / well / segment / row / rate-state / reduction順固定。
- stochastic components: なし。
- CPU/GPU: Kaggle private CPU version 1、GPU 0、internet offで完了。
- input SHA: exp209 / exp226 / saved fold kappa / raw well identityをhard guard実装済み。
- feature SHA: prefix residual、`mu_226`、`sigma_226`、fallbackをtruth join前にfreeze実装済み。
- gzip生成物: raw / decompressed / logical SHAを分離して記録する。
- prediction SHA: Stage 0AはHMM predictionを生成しないため非該当。
- model / submission SHA: 非該当。
- deterministic anchor: false。
- bootstrap: 実行configのloose / embedded parityとSHAをpush前に確認済み。

## 次のアクション

exp495はHMM実装前にfail-closedで完了。保存artifactだけの原因分解案をP4として残し、
現行P1/P2候補より優先しない。

## 2026-07-31 Stage 0B明示overrideとpush前実行量

ユーザー指示`Stage 0Bへ進んでください`を、Stage 0A mechanism gate FAILによる
停止条件の明示overrideとして記録した。Stage 0AをPASSへ再分類せず、事前登録した
fixed32条件を変更せずにStage 0Bだけを実装・Kaggle private CPU実行する。

| 項目 | 数 |
| --- | ---: |
| scientific variant | 1 |
| fixed32対象well | 32（persistent 16 / matched control 16） |
| reporting fold | 5 |
| candidate HMM well-run | 32 |
| parent/control HMM再実行 | 0 |
| fitted ML model / LightGBM config / trained fold / booster | 0 / 0 / 0 / 0 |
| PF / Beam / GPU | 0 / 0 / 0 |

比較には保存済みexp209 / exp355 predictionを使う。prefix window、sigma式・floor、
temperature、threshold、activation gate、emission、rate/TVT grid、blend、selector、PFは
変更しない。Stage 0Bの結果にかかわらずStage 1へ自動昇格せず、別承認を必要とする。

Stage 0B実装内容:

- fixed32 manifestはprediction freeze前に`well,prefix_rows,suffix_rows`だけを読む。
- Stage 0Aと同一のouter-fold donor replay、tail128 centered MAD、`mu_226` / `sigma_226`式を
  fixed32だけに再生する。
- exp209 absolute rate 41-state HMMのrate transition destinationへGaussian factorを掛け、
  各source-rate行で正規化する。position transitionは`r_j*ΔMD-ΔZ`のまま。
- uniform factorではparent posterior meanと`1e-10 ft`以内で一致するsynthetic parityを実行する。
- 32 candidate predictionとlogical SHAが全てfreezeした後だけrole、episode、suffix truth、
  保存済みexp209 / exp355比較を読む。
- Stage 0B gateはall32 / persistentのexp355比、matched controlのexp209比、fold、episode、
  by-well p95 / worstを事前閾値のANDで判定する。

push前検証:

```text
pytest: 22 passed
ruff F821/F401/F841/E9: PASS
py_compile: PASS
Jupytext --to ipynb --test: PASS
strict experiment validation: PASS
```

- loose / packaged config SHA: `2775534de9be6a2d9714ffac66115fbd7d8934885c26c5ab1ae964df6b2e6524`
- canonical Notebook SHA: `a1358e2d53b136de45f07d97c747759b3e1e4e43b366520175caceebc3500fd4`
- bootstrap済みpackage Notebook SHA: `b941a7f4a49abba3db03a7835bcd5f85c2bd5d58d94ed90d3bd76e19dd138c14`
- package input: exp226 v1、exp209 v3、exp355 v2、fixed32 manifest、persistent episodes
- 初回pushはKaggleのkernel source 1MB上限で拒否された。Notebookはself-containedで
  `src/` importが0のため、未使用の共通`src/` bundleだけを`--no-src`で除外し、
  packageを`781,010 bytes`へ縮小した。科学ロジック・config・依存artifactは不変。

Kaggle version 2は32/32 candidate HMM（156,088 rows）をfreezeまで完了したが、
post-freeze fixed32 identity照合が同値の`int32` / `int64` dtype差を
`DataFrame.equals`で不一致と判定しERRORになった。truth、role、episode、saved baseline、
gate readoutより前の実装上の比較バグで、科学式・予測・実行量は変更しない。
列順・well文字列と、整数へ正規化したprefix/suffix row値を比較するguardへ修正し、
同一Stage 0B条件で再実行する。

Kaggle version 3も32/32 candidate HMMをfreezeまで完了したが、post-freezeの
saved exp209 loaderに`local`キーのない設定を参照するpath解決バグが混入しており
`KeyError`でERRORになった。candidate prediction、科学式、gate閾値は不変。
exp209は従来どおりkernel source candidatesだけから解決し、bootstrap asset用の
`local`候補はfixed32 manifest / persistent episodeにだけ適用するよう修正する。

再実行前のpost-freeze loader監査で、Kaggle上のexp355 `metrics.json`は保存prediction
SHAをtop-level `prediction_sha256`に持つことを確認した。ローカル記録用metricsの
`sha256.prediction_logical`を参照していた実装を、実際のKaggle source schemaへ固定した。
値は事前登録済み`634303f0...e21`と同じで、baseline prediction自体は不変。

`kaggle kernels output --file-pattern`でpost-freeze評価に必要なexp209 prediction、
exp355 prediction / metricsだけを`/tmp`へ取得してloader chainを監査した。fixed32
156,088 rows / 32 wells、persistent 25 episodes、truth late-join、well / episode metrics、
全Stage 0B gate計算まで完走し、pre-freeze truth / role / episode readは0だった。
これにより次回run前にpost-freeze path・schema・identity・gate keyを実ファイルで確認済み。

## 2026-07-31 Stage 0B Kaggle実行結果

- kernel: `kentookumura/exp495-uncertainty-weighted-rate-obs-hmm-train`
- version / id_no: `4` / `129285050`
- private / CPU / internet off、status `COMPLETE`
- runtime: `923.702763044 sec`、candidate HMM合計`784.397361624 sec`
- 1 scientific variant、32 HMM well-runs、156,088 rows、5 folds
- parent/control再実行、model、booster、PF、Beam、GPU: すべて0

RMSE:

| scope | candidate | baseline | candidate差 | 判定 |
| --- | ---: | ---: | ---: | --- |
| all32 vs saved exp355 | 13.069256677 | 10.677951387 | +2.391305290 | FAIL |
| persistent vs saved exp355 | 16.344367163 | 14.200976371 | +2.143390792 | FAIL |
| matched control vs saved exp209 | 8.454838422 | 3.428436286 | +5.026402136 | FAIL |

改善foldは2/5、persistent episode SSE reductionは`0.069667260 < 0.10`、
by-well p95は`+16.564281738 ft > +0.25`、worstは`+23.911032044 ft > +2.0`。
mechanism gateは7/7 FAILした。

technicalはfixed32 identity、finite coverage、truth/role/episode pre-freeze read 0、
uniform-factor parent parity `4.58e-17 ft`、transition row sum `1.40e-14`、runtime、
RSS `1.305180 GB`をPASSした。posterior normalizationは
`5.6292e-06 > 1e-06`の1件だけFAIL。mechanismも全FAILのため数値許容差を救済しない。

主要logical SHA:

- scientific contract: `2760e9a5b96a5bd8eacd0ca329ebceac0c89555e74bf3cc7e7cb3180f4d98313`
- input manifest: `22d6ca0764661b4d8faeab03a7827ce060131f546ec7290cffff0462ca6f25f4`
- prefix uncertainty: `2e36d48081d5bd851d6a8e49a777c291af124a5dddedb5830fc1d4fb9fb46a37`
- rate schedule: `e31f048aeb8356a894d794d6c0cb1730006ebbe34c8282074a26ae0c445e8b59`
- prediction: `e550b0cc7fadfad38a4f0606f36eacdb3dc189c63029adcc355dde59bc17e84e`
- truth-late readout: `e1d33c68ffd77030f3b35dea9b1df41abff44d35cce7339ce43a6f2fd53ff382`

実行時SHA:

- config: `e5cc055fbae1e752747cf83aac1c1f7e86359fdf01945601d970eb546d3fdd23`
- canonical Notebook: `a345a7216ac229cfd092942b82f413a717c5e1d0c13b1a5ee817238b09c1958d`
- packaged Notebook: `ebc0defc6fad68342984dbfe2c7e0669b40bb8100c97739e862d379fdb0dc2e4`

decisionは
`close_without_sigma_window_scale_temperature_emission_grid_blend_selector_or_pf_rescue`。
`run_stage_0b`をfalseへ戻し、Stage 1へ自動昇格しない。
