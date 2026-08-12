# exp350_exp345_bidirectional_gr_affine_smoother セッションノート

## 目的

exp345で禁止していたfuture raw GR利用を、閉鎖済みexp345のpost-hoc救済ではなく独立実験として事前設計する。exp345 causal affineのpooled gainを保ちつつ、400/773 wells悪化とworst`+9.354827 ft`をfixed-interval bidirectional smoothingで抑えられるかを検証可能な状態にする。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage_0_failed_closed`
- 親: `exp345_exp209_time_varying_gr_affine_calibration_hmm`
- CV / LB: `14.367548324` / 未提出
- Notebook: compact self-contained trainを正規trainへ採用。inferenceはscaffold placeholderのまま
- 科学実装、正規Notebook採用、Kaggle Stage 0: 完了。technical PASS / scientific FAIL。Stage 1、inference、submissionは不適格・未実施

## 2026-07-22 設計作成

ユーザーの「これのバックログ、実験ディレクトリ、steeringを作成して設計を確定。実装はまだ」という依頼をdesign-onlyの明示承認として、次を作成した。

```bash
make new-steering EXP=exp350_exp345_bidirectional_gr_affine_smoother
make new-exp EXP=exp350_exp345_bidirectional_gr_affine_smoother
```

- exp350を次の未使用番号として採番した。
- exp345は`stage_0_full_failed_closed`のまま維持し、version 2成果物をimmutable controlへ固定した。
- 単一変更を、exp345 forward EKF recordに対する1回のfixed-interval extended RTS backward passへ限定した。
- future raw GRは推論入力として許可し、future/true TVTとscore roleはprediction freeze前に禁止した。
- saved parent/causal predictionをcontrolに使い、Stage 0のcontrol HMM再実行を0へ固定した。
- process noise、initial fit、observation、exp209 HMM parameter、decoderはexp345から変更しない。

## 設計したStage 0

- mask: exp345と同じlast-640、773 wells、494,720 score rows。
- candidate: `one_pass_bidirectional_rts_affine_schedule_on_exp209` 1本。
- forward / smoother: 773 / 773 wells。
- new exact-HMM: 773 well-runs。
- parent / causal control HMM rerun: 0 / 0。
- LightGBM config / trained fold / booster: 0 / 0 / 0。
- PF / Beam / GPU: 0 / 0 / 0。
- CPU、float64、GPU/internet off、runtime hard gate 8.5時間。

Technical gateはexp345 artifact SHA、forward schedule parity`1e-10`、saved metric parity`1e-10`、terminal identity、covariance PSD/contraction、scale clip`<=1%`、494,720 rows / 773 wells / 773 HMM、全finite、runtimeをAND条件にした。

Scientific gateはmasked parent比`>=0.05 ft`、causal比`>=0.02 ft`、両baseline比4/5 folds、hidden-like spatial/typewell-purged 2面の存在と非悪化、parent比by-well median/p95`<=0`、worst`<=+0.25 ft`、boundary p95`<=3 sigma`をAND条件にした。

GR reconstruction NLLは、smootherがcurrent/future GRを入力にするためin-sampleであり、診断専用とした。

## 再現性メモ

- seed policy: RNGなし、fold/well/row/forward/backward順固定。
- stochastic components: なし。
- CPU/GPU: canonical CPU float64、GPU/internet off。workers 2 / Numba threads 2を開始契約にする。
- input: exp345 promotion/prediction/schedule/process-noise/mask/metricsの既知SHAをhard preflightする。
- freeze: base path、forward state/covariance、smoothed state/covariance、schedule、numerical audit、predictionをtruth/role join前にSHA化する。
- gzip: raw SHAとdecompressed logical content SHAを分け、後者を主証拠にする。
- model / submission SHA: 非該当。
- deterministic anchor: 別承認のrerunでlogical SHA parityを確認するまでfalse。
- Kaggle bootstrap: package承認後にembedded config/source、kernel sources、CPU/GPU/internet metadataを照合する。現在はpackageしない。

## 既存候補との優先順位

exp345は平均改善を示したがtail failureが強いため、exp350を低・P3とする。現行P1のexp349/exp340、P2のexp346を追い越さない。exp338とはexp345の独立兄弟であり、exp350のdependencyやblend相手にしない。

## 次のアクション

strict実験検証とKaggle package監査を通し、Stage 0を同じcanonical kernelへ1 versionだけpushする。Stage 0全gate PASS時もStage 1は自動実行しない。

## 2026-07-23 Stage 0実装・push前監査

ユーザーの「実行してください」を、固定済みStage 0の科学実装、正規Notebook採用、Kaggle package/push/runの明示承認として記録した。Stage 1、inference、submissionは承認範囲外である。

- scientific variant: `one_pass_bidirectional_rts_affine_schedule_on_exp209` 1本。
- forward filter: 773 wells。exp345 causal scheduleとの`1e-10` parity後だけ平滑化する。
- bidirectional RTS smoother: 773 wells、1 backward pass。
- new candidate HMM: 773 well-runs。
- parent / causal control HMM rerun: 0 / 0。exp345 version 2保存predictionをSHA固定して利用する。
- LightGBM config 0、学習fold 0、booster 0、PF 0、Beam 0、GPU 0。
- Kaggle CPU、internet off。runtime hard gateは`8.5 h`。
- Stage 1、inference、submission、parameter/grid、blend、row/well gate、post-hoc救済は無効のまま。

実装Notebookは親exp345のcompact self-contained 11章構成を維持し、保存controlのSHA preflight、forward state/covariance再生成、forward parity、fixed-interval extended RTS、candidate HMM、truth/roleのlate join、technical/scientific gate、生成物SHAをセル上で追える。親は2,189行、exp350は約2,800行で、同一exp helper importと`__file__`依存はない。

静的検証:

```bash
.venv/bin/python -m py_compile experiments/exp350_exp345_bidirectional_gr_affine_smoother/exp350_exp345_bidirectional_gr_affine_smoother_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp350_exp345_bidirectional_gr_affine_smoother/exp350_exp345_bidirectional_gr_affine_smoother_compact_selfcontained_train.py experiments/exp350_exp345_bidirectional_gr_affine_smoother/tests/test_exp350_exp345_bidirectional_gr_affine_smoother.py
.venv/bin/pytest -q experiments/exp350_exp345_bidirectional_gr_affine_smoother/tests/test_exp350_exp345_bidirectional_gr_affine_smoother.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp350_exp345_bidirectional_gr_affine_smoother/exp350_exp345_bidirectional_gr_affine_smoother_compact_selfcontained_train.py
```

- `py_compile`: PASS。
- ruff: PASS。
- 専用test: `4 passed`。
- Jupytext round-trip: PASS。
- ローカルfull notebook実行: 未実施。初回実行先はKaggleとする。

Kaggle canonical kernelは、親exp345で長いslugがSaveKernel 400になった履歴を踏まえ、意味を保って`kentookumura/exp350-bidirectional-gr-affine-smoother-train`とした。titleは`exp350 bidirectional GR affine smoother train`でslugと一致させる。別slugは作らない。

### Stage 0 version 1 push

```bash
make prepare-kaggle-notebooks EXP=exp350_exp345_bidirectional_gr_affine_smoother EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp350-bidirectional-gr-affine-smoother-train --title 'exp350 bidirectional GR affine smoother train' --run-on-push --strict"
make push-kaggle-train EXP=exp350_exp345_bidirectional_gr_affine_smoother
```

- `Kernel version 1 successfully pushed`を確認した。
- canonical kernel: `kentookumura/exp350-bidirectional-gr-affine-smoother-train`。
- URL: `https://www.kaggle.com/code/kentookumura/exp350-bidirectional-gr-affine-smoother-train`。
- Stage 0だけを実行する。別slug、追加version、Stage 1は作成していない。
- push後にcanonical status `RUNNING`、remote `id_no=128274195`、CPU・internet off、kernel source 3件を確認した。
- ユーザー指示により継続監視を停止した。完了通知後に同じversion 1のstatus/logsを確認し、重複pushしない。

## 2026-07-23 Stage 0完了・閉鎖

ユーザーの完了通知後、canonical kernel version 1のstatusと完了ログを確認した。by-well failureと生成物SHAを正確に記録する必要があるため、Kaggle outputから小さいmetrics / gate / paired / by-well / numerical audit / summaryを一時領域へ取得した。大きいpredictionとsmoothed scheduleはリポジトリへ複製していない。

確認コマンド:

```bash
kaggle kernels status kentookumura/exp350-bidirectional-gr-affine-smoother-train
kaggle kernels logs kentookumura/exp350-bidirectional-gr-affine-smoother-train
kaggle kernels output kentookumura/exp350-bidirectional-gr-affine-smoother-train -p /tmp/exp350-stage0-output.EZIAGW --file-pattern '(metrics\.json|promotion_gate\.json|paired_metrics\.csv|by_well_metrics\.csv|numerical_audit\.csv|summary\.json|runtime\.csv|scientific_contract\.json)$' --page-size 200 --force
```

実行scope:

- kernel version 1、id_no `128274195`、status `COMPLETE`。
- CPU、GPU off、internet off。
- Stage 0のみ。forward 773 + smoother 773 + candidate HMM 773/773。
- parent / causal control HMM再実行0 / 0。
- LightGBM config / trained fold / booster / PF / Beam / GPU: 0 / 0 / 0 / 0 / 0 / 0。
- prediction 494,720 rows / 773 wells、全finite。
- runtime `2749.356610 sec = 45.82 min`。

technical gateはPASSした。

- exp345保存成果物SHA、saved parent/causal metric parity、494,720 rows / 773 wells / 773 HMM、finite、posterior normalization、terminal identity、covariance PSD/contraction、scale clip 0%、runtimeが全PASS。
- forward schedule parity最大差`7.070433e-11`は上限`1e-10`以内。
- parent / causal metric parity差はともに0。

scientific gateはFAILした。

- masked exp209 parent `14.501047783 → 14.367548324`、`+0.133499460 ft`、5/5 folds、hidden-like 2面改善はPASS。
- exp345 causal `14.331542697 → 14.367548324`、`-0.036005627 ft`、2/5 foldsでFAIL。
- parent比403/773 wells改善、370/773 wells悪化。median delta `-0.008671745 ft`はPASSしたがp95 `+1.346426592 ft`はFAIL。
- worst `8995c945`はparent `14.255149`、causal `10.343780`、candidate `35.142523`。parent比`+20.887374 ft`、causal比`+24.798744 ft`でFAIL。
- boundary jump p95 `0.839762 sigma`、scale clip率0%なので、worstはclipやcovariance破綻の単純な実装異常ではない。
- future GRを見たGR reconstruction NLL `4.640710856`は事前契約どおり診断専用。

主要SHA:

- prediction decompressed: `5b66ee9cfa76fce320a5806849e49c5f711b6dbafde617567adc211cda4de3f1`。
- forward schedule decompressed: `682e75d6cbed11b96e3b687f66a5c851399ae39250d5147ec3544601892a72cb`。
- smoothed schedule decompressed: `3c5a4268ee08e94b79b7a8fb971b2a678a45cde9edae6743e67db8cb881ac88b`。
- freeze manifest: `57d7a6207225281c8e4b5a517db1c286f3682b3bffda4b05036516826e471df6`。
- promotion gate raw: `2f003dfd46cda4efcd25c1e2555bf68a77db78273342f2425809c5a109d94616`。

解釈と判断:

- forward parityと数値gateが成立したうえでcausalを下回ったため、implementation failureではない。
- fixed base pathに対するfuture GR smoothingが、一部wellの誤calibrationをprefix方向へ広く逆伝播し、exp345 causalのtailを抑えるどころかworstを拡大した。
- decisionは`stage_0_failed_close_without_rescue`。Stage 1、version 2、Q/rcond/clip/iteration/grid、causal blend、row/well gate、inference、submissionは行わない。
- backlogから完了済みexp350を削除し、同familyの救済候補は追加しない。独立P1--P2のexp340を優先する。

## Design-only検証

```bash
make validate-exp EXP=exp350_exp345_bidirectional_gr_affine_smoother
make update-summary
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp350_exp345_bidirectional_gr_affine_smoother --root .
```

- `config.yaml`: YAML parse PASS、route=`pf_beam`、status=`design_frozen_not_implemented`。
- `metrics.json`: JSON parse PASS。
- implementation/package/push/run/inference/submission flag: すべてfalse。
- strict experiment validation: PASS。
- experiment docs reviewer: core evidence categories present、exit 0。
- `experiment_summary.md`: exp350を追加済み。
- 科学実装source、Kaggle package、prediction、model、submission: なし。
