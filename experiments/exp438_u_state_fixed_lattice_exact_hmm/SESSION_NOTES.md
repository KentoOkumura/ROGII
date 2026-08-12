# exp438_u_state_fixed_lattice_exact_hmm セッションノート

## 目的

exp209 exact HMMの持続位置状態をTVT固定格子から絶対U固定格子へ変え、
連続運動学を固定したままZ由来の離散化位相だけを検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage0_fail_closed_v1`
- CV / LB: なし
- 実装承認 / Stage 0実行承認: あり / 消化済み
- compact train: 実装済み、正規notebookへ採用済み
- 正規notebook: Kaggle private CPU version 1で実行完了
- inference / submission: 無効

## 2026-07-29 設計セッション

ユーザー依頼:

```text
HMMの状態をTVTからUにする実験を行いたいです
```

設計上の整理:

- `U=TVT+Z`なので連続状態の座標変換だけならexp209と厳密に同値。
- rowごとにU格子を`TVT grid+Z_t`へずらす実装は科学的ablationにならない。
- 実験差を作るため、最後の既知Zでanchorした絶対U格子を全rowで固定する。
- 親TVT格子と同じstep、cell数、start indexを使い、state数とprior massを固定する。
- transitionは到着rateの`ΔU=r_t*ΔMD`、emission/readoutは`TVT=U-Z_t`。
- source/destination rateの台形積分は併用せず、固定格子座標だけを1-factorとする。

根拠と注意:

- exp435はrate履歴を捨ててmatched controlを大幅悪化させたため再開しない。
- exp408ではposition quantization biasはpersistent offsetの主因ではなく、
  誤ったrate posteriorに対するregularizerの可能性も示された。
- したがってquantization bias低下だけでは採用せず、fixed32 safetyと
  Stage 1 full OOFをAND判定する。

## 予定実行量

- scientific variant: 1
- Stage 0: 32 HMM well-runs
- Stage 1最大: 773 HMM well-runs
- reporting folds: 5
- parent control HMM再実行: 0
- model / LightGBM config / trained ML fold / booster: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`

この量は設計値であり、push前に再確認して本ファイルへ記録する。

## コマンドログ

```bash
make new-steering EXP=exp438_u_state_fixed_lattice_exact_hmm
make new-exp EXP=exp438_u_state_fixed_lattice_exact_hmm
```

設計文書、metadata、markdown-only notebook placeholderの静的検証だけを行った。
この設計セッションではtrain/inference、Kaggle package/push/runは未実行。

## 2026-07-29 Stage 0実装

ユーザーの`exp438を実装してください`を、compact self-contained Stage 0候補と
専用testの実装承認として記録した。既存の正規notebook置換、Kaggle
package / push / run、Stage 1、inference、submissionの承認には拡張していない。

実装内容:

- Jupytext percent形式の
  `exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_train.py`
  とfail-closed inference候補を作成した。
- exp209と同じjoint position/rate sum-product HMMをcoordinate-genericに実装し、
  candidateは固定U格子と`transition_coordinate_delta=0`を使用する。
- U格子は親TVT格子へ最後の既知Zを一度だけ加える。row adaptive regrid、
  re-anchor、interpolation transportはない。
- emissionは各rowで`GR_typewell(U_grid-Z_t)`、readoutは`E[U]-Z_t`。
- rate kernel、arrival-rate position mean、5-cell support、noise、momentum、
  initial prior、Gaussian emission、posterior mean/stdはexp209 contractに固定した。
- smoothed rate posteriorを保存し、同じposteriorで親TVT式とcandidate U式の
  position-kernel quantization biasを重み付けするtruth-free ledgerを実装した。
- fixed32 manifestはprediction前にwell/prefix/suffixだけ読み、role/foldは
  32 wellsのprediction/rate/transition SHA freeze後に再読する。
- suffix truth、persistent episode、exp408 cause、error metricも全freeze後だけ読む。
- technical/mechanism gateは全条件ANDで、1条件FAIL時はStage 1へ進めない。

数値contract:

- `TVT_state=U_state-Z_t`、`delta_TVT=delta_U-delta_Z`、
  emission direct lookup、`E[TVT]=E[U]-Z`を検査する。
- constant-Z synthetic parent parityを検査する。
- tiny joint HMMの全initial/suffix state pathを列挙する独立referenceと、
  log-likelihood、position/rate posteriorを比較する。
- exp209の独立`_hmm2_fb`とgeneric fixed-lattice kernelを回帰比較する。
- rate/position transition row sumとsmoothed posterior normalizationを検査する。

親compact比較:

- 科学的親exp209にはcompact self-contained版がないため、同じexp209 joint HMMを
  fixed32で実装したexp411 compact trainを構成参照にした。
- exp411 / exp438 train sourceは`2,255 / 2,785 lines`。
- どちらも9章構成で、exp438はnotebook-safe path/SHA、fixed32/saved parent、
  input preparation、route-specific joint HMM、numerical contract、
  target-free freeze、truth-late readout、gate/生成物、guarded orchestrationを持つ。
- 同一exp helper import、`__file__`、`prange`、薄い`main()` entrypointはない。

検証:

```bash
.venv/bin/python -m py_compile \
  experiments/exp438_u_state_fixed_lattice_exact_hmm/*compact_selfcontained*.py \
  experiments/exp438_u_state_fixed_lattice_exact_hmm/settings.py
.venv/bin/ruff check \
  experiments/exp438_u_state_fixed_lattice_exact_hmm/*compact_selfcontained*.py \
  experiments/exp438_u_state_fixed_lattice_exact_hmm/settings.py \
  experiments/exp438_u_state_fixed_lattice_exact_hmm/tests/test_exp438_u_state_fixed_lattice_exact_hmm.py --select F821
.venv/bin/pytest -q experiments/exp438_u_state_fixed_lattice_exact_hmm/tests/test_exp438_u_state_fixed_lattice_exact_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp438_u_state_fixed_lattice_exact_hmm/\
exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_train.py \
  experiments/exp438_u_state_fixed_lattice_exact_hmm/\
exp438_u_state_fixed_lattice_exact_hmm_compact_selfcontained_inference.py
make validate-exp EXP=exp438_u_state_fixed_lattice_exact_hmm
make validate-template
```

- py_compile: PASS
- Ruff F821: PASS
- 専用test: `12 passed`
- Jupytext train / inference round-trip: PASS
- strict experiment validation: PASS
- template validation: PASS
- ローカルnotebook実行、Kaggle package / push / run: 未実施

## 再現性メモ

- `docs/06_reproducibility.md`確認済み。
- RNGなし。well / row / U state / rate / edge順を固定する。
- input、fixed32、coordinate contract、transition quantization ledger、
  prediction、posterior/rate readout、gate reportのlogical SHA保存を実装済み。
- gzipはdecompressed content SHAを主証拠にする。
- 初回runをdeterministic anchorとしない。同一設定rerunのtransition /
  prediction SHA一致を必要とする。
- input / prediction SHAは実装済みだが未実行のため未生成。

## 次のアクション

1. compact候補の正規notebook採用は別承認まで行わない。
2. Kaggle package/push/runは別承認とする。
3. Stage 0全gate PASS後だけStage 1を別承認で検討する。
4. inference / submissionはStage 1採用判断と別承認まで無効のままにする。

## 2026-07-29 Stage 0実行承認

ユーザーの`実行してください`を、直前に実装したexp438 compact候補の
正規notebook採用と、Kaggle private CPU fixed32 Stage 0の
package / push / run承認として記録した。

承認範囲外:

- Stage 1 full OOF
- inference実行
- submission生成・提出
- grid/noise/rate/emission/anchor/phaseの追加variant
- exp209 parent HMMの再実行

push前の実行量:

- active scientific variants: `1`
- Stage 0 target wells / HMM well-runs: `32 / 32`
- reporting folds: `5`
- saved exp209 parent HMM reruns: `0`
- LightGBM configs / trained ML folds / boosters / fitted ML models:
  `0 / 0 / 0 / 0`
- PF / Beam / GPU runs: `0 / 0 / 0`
- runtime: Kaggle private CPU、internet無効、Numba thread `1`
- submission生成: `0`

canonical kernel:

- id: `kentookumura/exp438-u-state-fixed-lattice-exact-hmm-train`
- title: `exp438 u state fixed lattice exact hmm train`
- parent kernel source: `kentookumura/exp209-joint-exact-parity-train`

credential checker:

- API token: 未設定
- Kaggle CLI OAuth: 利用可能
- legacy credential: 利用可能
- credential実値は記録していない

採用前SHA:

- compact train notebook:
  `9074dc2c381e1ed2d8bf094e24846d0ee07845b3212ee87bed6c3ba02db4d4d0`
- 旧正規train placeholder:
  `632cc22354e6d492a57172ea4d3166fff493c981999d9aa69f5379f29c9c4ba4`
- compact inference guard:
  `df31b4013b136e93aa4c0c39a8b1d16adfa4e114693f97fc990819fbe632cbd6`
- 旧正規inference placeholder:
  `aee2a6eb31ed33fb224af354f3c2835c930a1584f2c3bca3152c725e1dba9ecd`

正規notebook採用時の確認:

- train source末尾を`if __name__ == "__main__"`でguardし、pytest import時は
  Stage 0を起動せず、Kaggle notebook実行時だけ`run_stage0`へ入る形にそろえた。
- guard修正後も専用test `12 passed`、Ruff F821、py_compile、
  Jupytext round-trip、strict experiment/template validationを再度PASSした。
- final compact / canonical train SHA:
  `56ec938fff150b46e2ce2546f33ca2ab33cf0ebe417ec418af19f8329f8f5e9b`
- final compact / canonical inference SHA:
  `df31b4013b136e93aa4c0c39a8b1d16adfa4e114693f97fc990819fbe632cbd6`

strict Kaggle package監査:

- prepare:
  `make prepare-kaggle-notebooks EXP=exp438_u_state_fixed_lattice_exact_hmm EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp438-u-state-fixed-lattice-exact-hmm-train --title 'exp438 u state fixed lattice exact hmm train' --run-on-push --strict --no-src"`
- package profile: self-contained `--no-src`
- packaged notebook SHA:
  `83f74c066c94ff09a356994c00f9b1f9c2b64d8f4097dd337dea0b82114c5abe`
- packaged `kernel-metadata.json` SHA:
  `20dd29ed99ca267a1dbf645e1bbab412316e8ad7d9c07990a63ac56d0b63b4ba`
- packaged `config.yaml` SHA:
  `4070fb59c0a4e6402da7ed9edbae880e3c0bfb4c285705877ce3c2f2ed1396ba`
- metadata: private、CPU、internet無効、`run_on_push=true`
- competition source: `rogii-wellbore-geology-prediction`
- kernel source: `kentookumura/exp209-joint-exact-parity-train`
- bootstrap内で次の3入力を埋め込み、展開後SHA照合を行う:
  - `assets/stage0_fixed32_manifest.csv`
    (`fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`)
  - `assets/persistent_offset_episodes.csv`
    (`031067fa77c195b77920a0997401310fbdd16532a2d0e99a9c3b5044de28913c`)
  - `assets/exp408_hmm_message_rate_basin_audit_episode_summary.csv`
    (`b230ffc759e6ee4891f22809b3f3c8a8796681fb461ec0b7215b94a352bf0ab0`)

Kaggle push:

- version: `1`
- pushed at: `2026-07-29 12:37:51 UTC`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp438-u-state-fixed-lattice-exact-hmm-train`
- status at handoff to monitor: running

## 2026-07-29 Kaggle Stage 0 version 1結果

- Kaggle status: `COMPLETE`
- kernel id_no: `129056676`
- completion observed: `2026-07-29 13:03:10 UTC`
- notebook metrics created: `2026-07-29T13:02:16.867261+00:00`
- scientific variants / HMM well-runs / reporting folds: `1 / 32 / 5`
- parent HMM rerun / ML model / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0`
- Stage 0 elapsed: `1452.118439 sec`（約24分12秒）
- candidate HMM: `1403.666002 sec`
- Stage 1 runtime projection: `33907.306853 sec`
  （上限`30600 sec`を`3307.306853 sec`超過）
- peak RSS: `1.133476 GiB`

technical gate:

- `runtime_projection`だけFAIL。
- coordinate identity、constant-Z parent parity、brute-force reference、
  transition/posterior normalization、finite coverage、truth-late ledger、
  readback SHA、RSSを含む残り20項目はPASS。
- constant-Z prediction max abs: `5.4569682106375694e-12 ft`
- brute-force maximum abs: `2.9459618633431717e-08`
- transition row-sum max error: `3.3306690738754696e-16`
- posterior normalization max error: `1.5543122344752192e-15`

mechanism gate:

- 7項目すべてFAIL。
- posterior-weighted quantization bias:
  parent `3658.937047 ft`、candidate `5253.499371 ft`、
  reduction `-0.435799`（43.58%悪化、必要値`>=0.10`）。
- forward-cause episode SSE:
  parent `10827465.785976`、candidate `34084417.666782`、
  reduction `-2.147959`（必要値`>=0.05`）。
- persistent episode SSE:
  parent `13363710.665031`、candidate `123511976.228704`、
  reduction `-8.242341`（必要値`>=0.03`）。
- persistent improved wells: `2 / 16`（必要`>=10`）。
- persistent improving folds: `0 / 5`（必要`>=4`）。
- matched control pooled RMSE:
  parent `3.428436 ft`、candidate `46.748911 ft`、
  delta `+43.320475 ft`（許容`<=+0.02 ft`）。
- control by-well delta p95: `+72.480990 ft`
  （許容`<=+0.25 ft`）。
- persistent episode SSEは全5 foldsで悪化した。

判定:

- `stage0_all_gates_pass=false`
- `stage1_eligible_for_separate_approval=false`
- `stage0_fail_closed`
- 事前登録どおり、grid phase/anchor/step/band、noise、rate、
  emission、blend/selectorによる同一fixed32 rescueは行わない。
- Stage 1 full OOF、inference、submissionへ進まない。
- fixed32はmechanism preflightであり、CVやpromotion evidenceではない。

artifact SHA:

- Kaggle `metrics.json`:
  `7118b009a518aec5bf0bc2916bc38a273243202cc701ccf6912d61dc4f9f2322`
- gate report:
  `ed4a3745acafe29e97e031442054d92c67137441ac181def8809f39468233ff2`
- numerical contract:
  `124a5c75cde6a22964ee217ad8323f97a7ecee0d934c9bd19194d734a02ae2f1`
- episode metrics:
  `0c59103d8e06a2ee3b8b2a105130e1154b664b31ef8ab30d878261b9261a5695`
- well metrics:
  `6af93b427f8272f13c818156e2fbce91657a0847cca8322a762c8cfde777d489`
- prediction logical:
  `6a2096746b4b98bf192be5180b8f3763e528262b91bcedd8301ece927993753d`
- rate readout logical:
  `9e3359964c46fdf00560b293ef52eed2bd97cf2003bfaff88b194010db5f0d8b`
- transition ledger logical:
  `24ee57208298466bb215eb763f4073b2ab2f62e198d40be21a1105c2d3d68806`

再現性:

- 初回runをdeterministic anchorにはしない。
- FAIL判定を確定するためのrerunは不要で、parent/controlも再実行していない。
- 現行prediction/transition SHAはnegative mechanism evidenceの参照として保持する。

post-run local package lock:

- remote Kaggle version 1は変更していない。
- ローカル`kaggle/train`だけを再生成し、`run_on_push=false`、
  `execution.run_hmm=false`、`runtime.run_approved=false`へロックした。
- locked packaged notebook SHA:
  `687cb5d973fa341fe833dc65acaa1022d0a2f94e254812cd4da2da70d19ffdbc`
- locked `kernel-metadata.json` SHA:
  `73226f7339f0c20a972152287e6e84d5556f55add62fb96d63d4aa25d737a8d0`
- locked source / packaged `config.yaml` SHA:
  `b5eadaae124af0840ab374063c83f10ebbcf89c5d7860befdf74a38e446901f1`
  （byte-identical）
- executed version 1 package SHAは前節のpre-push監査値を正とする。
