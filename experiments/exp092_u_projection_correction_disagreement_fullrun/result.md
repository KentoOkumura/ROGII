# exp092_u_projection_correction_disagreement_fullrun 結果

## 状態

Kaggle train v1、OOF delta guard、user-requested inference、code submission 完了。正式 pooled OOF、by-well、bucket、feature importance、model manifest、OOF prediction、exp073 / exp077 align guard、Public LB を取得済み。

## 仮説

exp085 の log-derived 評価では、`u_projection_correction_plus_disagreement` が control より改善した。exp085 は全 variant x 3 model の実行で timeout したため、exp092 では最有望 variant だけを完走させ、正式 pooled OOF と監査生成物を得る。

## 評価方針

exp072 deterministic full replay train cache と exp073 LightGBM config family を固定し、GroupKFold by well で `u_projection_correction_plus_disagreement` の 3 model と `lgb_mean` を評価する。

比較基準:

- exp085 log-derived control / `lgb1`: mean fold RMSE 9.534549。
- exp085 log-derived selected / `lgb1`: mean fold RMSE 9.291006。
- exp077 policy OOF: RMSE 9.470514801。

## 結果

Kaggle train v1 は `kentookumura/exp092-uproj-corr-disagree-train` version 1 で完了した。runtime は 13,563.193 秒、入力は exp072 full replay cache 3,783,989 rows / 773 wells、base 196 features に U-projection correction / disagreement features を加えた 240 features。

| model | pooled OOF RMSE | prediction SHA |
| --- | ---: | --- |
| `lgb1` | 9.322479896 | `dd631f28f3cfc6da3cab1ec3e939bb7185c5546e5b77c33e4425ec8080ef42e0` |
| `lgb2` | 9.338192405 | `85d0c7fd1a7482299fdf3a91527b3bea2656ac64529f2e5d778b961e6d271c56` |
| `lgb_mean` | 9.343064066 | `adbadb268707032f0ce3cd493a7a0ee81d43269574069c2c36826dd387c56a3b` |
| `lgb0` | 9.533126438 | `b98e93ae49283204cec06b12256b05326c2f7dff10a388c30e1944f0d45fc89b` |

最良は `lgb1` の 9.322479896。exp077 policy OOF 9.470514801 から -0.148034906、exp073 raw anchor 9.526374749 から -0.203894854 改善した。exp085 の log-derived `lgb1` 平均 9.291006 より正式 pooled は悪いが、改善方向は再現した。

distance bucket では `lgb_mean` が 0-50 ft で RMSE 1.154526、50-100 ft で 1.443261 と near-prefix を壊していない。一方で全体最良 `lgb1` の 1000+ ft は RMSE 10.229080、`lgb_mean` は 10.255952 で long-tail が主誤差源のまま。

worst well は `86454a6f` が RMSE 57.69 前後、`1b1eba53` が 41-44、`fb03ae90` が 39-40 と大きい。exp073 / exp077 との by-well delta guard は後続の worst-well gating / regression guard の入力として扱う。

上位重要特徴には既存 surface の `spatial_knn_dist`、`frac`、`slp_b_d_50`、`dense_std`、`dz` が並ぶ一方、追加 U-projection feature では `uproj_likpf_mean_resid_mad` と `uproj_pf_ancc_resid_mad` が top 30 に入った。

OOF predictions gzip SHA は `67617007921e762868063c27e1bfa6156622c4357b3c922b6b37eda3c21235b9`、decompressed content SHA は `6dc3d53d6cb5621b86360929d638f2a5d853c58ea3e7a53d3da86e614e5f2f69`。model manifest は 15 fold models を記録している。

## OOF delta guard

`artifacts/oof_delta_guard/` に exp073 `lgb_mean`、exp077 `longtail_likpf_tiny_gate_w006`、exp092 predictions の align guard を保存した。local の exp072 train feature cache gzip が 0 byte だったため、near/long bucket は `id` 末尾の tail rank fallback で見ている。

aligned OOF 3,783,989 rows / 773 wells で、exp092 `lgb1` は exp077 policy 9.470514801 に対して 9.322480157、delta -0.148034643。exp073 `lgb_mean` 9.526374826 に対しては delta -0.203894669。`lgb_mean` も exp077 から -0.127450731、exp073 から -0.183310756 改善した。

long-tail `1000_plus` は 3,783,839 rows で `lgb1` が exp077 から -0.148037816、exp073 から -0.203898884 改善した。一方、near-row 0-250 はこの OOF surface に行がなく、tail-rank 500-999 も 149/150 rows だけで +0.10 程度悪化したため、near-row guard は pass ではなく inconclusive とする。

by-well は exp077 比で 459 wells 改善 / 314 wells 悪化。最大悪化は `b8c49c1a` +4.164460、次いで `3417285d` +3.590497、`f074d277` +3.352546、`f9fc81aa` +3.007717、`86454a6f` +2.951191。最大改善は `389ae58f` -5.348768。max well regression は warning threshold 0.25 を大きく超える。

path continuity は全体崩壊ではない。`pred_exp092_lgb1_step_abs_p95` は well 平均 0.327492、全 well p95 0.526094、最大 0.903320。`pred_exp092_lgb1_step_abs_max` は p95 4.116602、最大 10.460938で、ge10 spike は 1 件、ge25 は 0 件。exp077 との差分 correction step p95 は全 well p95 0.456641、最大 0.857471、correction step ge5 は 23 件。

## Public LB

| ref | Public LB | note |
| --- | ---: | --- |
| `53927479` | 8.350 | user-corrected exp092 submission |

Kaggle submission description was blank. The user corrected `ref=53927479` / Public LB 8.350 as exp092. Local submission output SHA is not recorded in this repo.

Public LB 8.350 improves exp077 8.611 by -0.261 and exp098 8.441 by -0.091, and becomes the ML route submitted anchor. It remains weaker than the ensemble route anchor exp082 7.601.

## Visible-test guard

`worst_well_rawtest_guard.py` を追加し、OOF worst-regression wells と exp092 inference prediction を接続する target-free guard を実装した。ただし、このコンペは Code Competition 形式であり、通常の Kaggle notebook 実行で読める test は exposed sample / visible test である。LB 採点時の hidden test は code submission rerun 時に差し替えられるため、この guard は hidden LB test の検査結果ではない。

この guard で確認できるのは、通常 kernel 上の visible test に対する schema parity、projection summary parity、prefix anchor parity、well-level prediction step、optional exp073 / exp077 比 correction だけである。hidden test 側で見たい事象は、提出 notebook 内に assert を仕込み、submission が通るか落ちるかで間接的に見る必要がある。

出力先は `artifacts/worst_well_rawtest_guard/`。`--self-test` は合成データで PASS し、`py_compile` / `ruff check` / `ruff format --check` も PASS した。

Kaggle guard v2 は `kentookumura/exp092-worst-well-rawtest-guard` version 2 で完了し、local output は `kaggle/output/guard_v2/artifacts/worst_well_rawtest_guard/` に保存した。status は `visible_test_completed_pass`。visible test は 14,151 rows / 3 wells、schema parity は 240 features 完全一致、raw prefix anchor と exp092 `last_known_tvt` の最大差は 0.0。warning wells は 0。projection summary parity は sources 5、max abs ratio 1.470640。exp092 inference prediction decompressed SHA は `c863d55011690f4cc7f96e1a814619c5cf68ba4b9b83ed038756ccb25e302c5e`、guard prediction SHA は `9882797f249273e6f1911e58b6a8ad8b385b91cea3d4dfebf9e03f47b9d07332`。

v1 は generic `submission.csv` glob により mounted exp092 submission を optional `exp077_submission` と誤検出したため、修正して v2 を正とする。v2 では exp077 baseline は未指定扱いで、exp092 raw-test continuity / schema / projection parity のみを判定した。

## Hidden assert probe

実験目的を hidden LB test の直接観測ではなく、Code Competition の submission rerun に opt-in assertion を仕込み、pass/fail だけで hidden 側の条件を間接観測する形に修正した。`run_saved_model_inference()` は `inference.hidden_assert_probe` を受け取り、デフォルトでは無効。probe 有効時も通常 kernel の visible test signature では skip できる。

assert 条件は次の通り。

- `non_visible_signature`: hidden probe として実行される入力が exposed visible test signature ではない。
- `sample_id_coverage`: `sample_submission.id` 全行に対して prediction が存在し、fallback rows が 0。
- `finite_predictions`: `last_known_tvt`、`pred_delta`、`pred_tvt` がすべて finite。
- `anchor_t0_abs_max`: raw well CSV から復元した prefix anchor `TVT_input` と `last_known_tvt` の最大差が `0.05` 以下。
- `known_prefix_rows_min`: 各 well に少なくとも 1 行の既知 `TVT_input` prefix がある。
- `well_step_abs_p95_max`: well 内 `pred_tvt` 隣接差分 p95 が全 well で `2.0` 以下。
- `well_step_abs_max_max`: well 内 `pred_tvt` 隣接差分最大値が全 well で `10.0` 以下。
- `pred_delta_abs_p95_max`: well 内 `|pred_tvt - last_known_tvt|` p95 が `100.0` 以下。
- `pred_delta_abs_max_max`: well 内 `|pred_tvt - last_known_tvt|` 最大値が `160.0` 以下。
- `pred_range_max`: well 内 `pred_tvt` range が `180.0` 以下。
- `near_prefix_delta_abs_p95_max`: 各 well の先頭 250 prediction rows で `|pred_delta|` p95 が `25.0` 以下。
- `near_prefix_delta_abs_max_max`: 各 well の先頭 250 prediction rows で `|pred_delta|` 最大値が `50.0` 以下。
- `near_prefix_step_abs_p95_max`: 各 well の先頭 250 prediction rows で `pred_tvt` 隣接差分 p95 が `1.5` 以下。
- `near_prefix_step_abs_max_max`: 各 well の先頭 250 prediction rows で `pred_tvt` 隣接差分最大値が `5.0` 以下。
- `projection_feature_finite`: U-projection / disagreement feature がすべて finite。
- `projection_correction_abs_p95_max`: projection correction / residual 系 feature の列別 abs p95 が `20.0` 以下。
- `projection_correction_abs_max_max`: projection correction / residual 系 feature の列別 abs max が `80.0` 以下。
- `u_disagreement_abs_p95_max`: PF/Beam/likelihood-PF U-space disagreement 系 feature の列別 abs p95 が `250.0` 以下。
- `u_disagreement_abs_max_max`: PF/Beam/likelihood-PF U-space disagreement 系 feature の列別 abs max が `500.0` 以下。

失敗時は `AssertionError("hidden_assert_probe_failed:<check_names>")` のみを投げ、hidden の行数、well 数、集計値、超過量は出さない。probe 有効かつ hidden context では summary / metrics / prediction artifact も redaction し、submission.csv だけを scoring 用に残す。

追加した `pred_delta`、near-prefix、projection/disagreement feature の条件が、exp092 の OOF worst-well 悪化、near-row inconclusive、U-projection 過補正懸念を hidden test 上で label-free に見るための主検証である。

2026-06-22 の code submission probe 結果:

- v2 `ref=53931397` / `exp092 hidden assert proxy probe v2` は Public LB 空で完了。UI/API は generic rerun exception しか返さないため、複数 assert のどれが落ちたかは不明。
- v3 `ref=53933465` / `exp092 probe sample_id_coverage v3` は Public LB `8.350` で完了。`sample_id_coverage` 単独 probe は hidden rerun で pass した。

したがって、v2 失敗原因は hidden sample ID coverage / fallback 0 ではない。残る未分解の候補は、prediction continuity、`pred_delta`、near-prefix、projection correction、U-space disagreement の各 proxy 条件である。

ただし、2026-06-22 に OOF guard 既存生成物で確認した限り、projection/disagreement 系を「悪化 well 固有の原因」として hidden assert する根拠は弱い。feature importance では projection correction group が importance share `0.1228`、U-space disagreement group が `0.1422` と使われている一方、既存 OOF の実予測補正量 proxy は悪化 top30 だけでなく改善 bottom30 でも大きく、RMSE 悪化との単純な正相関は見られなかった。

- `lgb1_minus_exp077_policy_correction_abs_p95`: all median `3.119`、悪化 top30 median `5.080`、改善 bottom30 median `6.966`、Spearman `-0.0859`
- `lgb1_minus_exp073_correction_abs_p95`: all median `3.216`、悪化 top30 median `5.326`、改善 bottom30 median `6.750`、Spearman `-0.1174`

そのため、`projection_correction_abs_*` / `u_disagreement_abs_*` の hidden assert は、現時点では worst-well 悪化検証ではなく broad sanity guard としてのみ扱うべきである。悪化検証として使うには、projection/disagreement feature 値を OOF well ごとに保存・集計する readout を先に作る必要がある。

この判断に基づき、exp092 の hidden assert probing は打ち切った。後続実験の参照元としてノイズにならないよう、`u_projection_correction_disagreement_fullrun.py` と inference notebook の実行パスから hidden assert hook / redaction branch を削除し、通常の saved-booster inference に戻した。probe 結果は履歴診断としてのみ扱う。

## 次の判断

train-side / long-tail は有望で、Public LB 8.350 も ML route 最良になった。一方で by-well 最大悪化が大きく、near-row は今回の OOF surface では十分に評価できない。visible-test guard v2 は hidden LB test の安全性を証明しないため、exp092 の採用根拠は Public LB 8.350 と OOF に置く。hidden assert probing は worst-well 懸念を直接検証できないため打ち切り、今後は OOF 側で worst-well mitigation / shrink / gate を検討する。
