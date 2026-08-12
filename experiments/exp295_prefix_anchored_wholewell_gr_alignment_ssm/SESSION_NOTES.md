# exp295_prefix_anchored_wholewell_gr_alignment_ssm セッションノート

## 目的

neighbor well dataなしのcomplete-well GR matchingを、prefix-conditioned learned emissionとfixed exact state-space decoderとして設計し、LB 5.xを狙う反証可能なStage A/B/C契約へ固定する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage A fold 0 version 3 runtime timeout・Stage B branch close
- CV / LB: なし / なし
- implementation approval: 2026-07-19ユーザー依頼「exp295を実装してください」
- Kaggle GPU push approval: 2026-07-19ユーザー依頼「実行してください」（Stage A fold 0のみ）
- runtime contract repair / version 3 approval: 2026-07-20ユーザー依頼「実行してください」
- inference / submission approval: なし / なし
- 現在のactive architecture / running neural model / 学習済みneural model / LightGBM config / booster / PF-Beam run: `1 / 0 / 0 / 0 / 0 / 0`

## 2026-07-20 Stage A Kaggle実行再開

- ユーザー依頼「実行してください」を、GPU枠解放後のStage A fold 0再実行指示として扱った。
- 実行契約は変更なし: `1 architecture x fold 0 x seed 42 = 1 neural model`、LightGBM 0、booster 0、PF/Beam 0、parent/control再学習0。
- canonical draft version 1をpullし、id_no `127896241`、private、GPU off、internet off、exp209 kernel sourceを再確認した。
- push前GPU quota: `17.59h used / 27.41h remaining / 45.00h total`、refresh `2026-07-25T00:00:00`。
- final canonical `kentookumura/exp295-prefix-gr-ssm-fold0-train`だけをT4/run-on-pushへ更新する。旧placeholder、Stage B、inference、submissionには触れない。
- T4/run-on-push packageを同じfinal canonicalへpushし、`Kernel version 2 successfully pushed`を確認した。
- run URL: `https://www.kaggle.com/code/kentookumura/exp295-prefix-gr-ssm-fold0-train`。
- Kaggle list上の開始時刻: `2026-07-20T00:16:58.457Z`。初期status: `KernelWorkerStatus.RUNNING`。
- 初期CLI logsは空。実行中logs空を失敗根拠にせず、別slugへの再pushは行わない。
- 通常pullは実行中version 2ではなく確定済みversion 1の`GPU off / machine_shape None` metadataを返した。明示version 2 pullは実行中のため403だった。Kaggle API lagとして記録し、version完了後に同slugをpullしてT4を確認する。
- version 2 の初期 status 時点の running neural model / 学習完了 neural model: `1 / 0`。その後 `ERROR` で停止し、現在の running / 学習完了は `0 / 0`。Stage B、inference、submissionは未実行。

### version 2失敗診断

- final status: `KernelWorkerStatus.ERROR`。実行は約223秒で停止し、完了epoch/model/予測/提出は0。
- 最初の意味のあるtraceback: `RuntimeError: 1478: true quantized path is infeasible under fixed exp209 grammar`。
- 分類: GPU/OOM/network/data pathではなく、hard-path structured CRF NLLと疎なtruth jumpのtraining objective/data契約不整合。CUDA必須guard、bootstrap、scientific/cost contract previewは通過した。
- 最小再現: `72dd8501 / pseudo_minus_256`、prefix end 1478、row 6223でtruth delta `-1.49 ft`、quantized shift `-5`に対し、固定exp209 grammarの同row union supportは`[-2, 2]`。target-in-grid rateは1.0。
- outer-train fit view全体のtruth-only audit: `469 / 1668 views`、`918 / 8,569,737 transition rows`が少なくともunion support外。約0.011%の疎なjumpだが、hard exact path likelihoodでは1行でも全view partitionが不可能になる。
- exp209 decoderのrate/position transition実装とのparity、`dm >= 1.0`、band/state step/rate spanは一致しており、decoder実装の転記bugではない。
- 固定decoder transitionを広げる修正はexp295契約違反。推奨はdecoderを変えず、structured label likelihoodをhard pathからstate step由来のGaussian observationへ変更すること。loss意味が変わるため再push前にユーザー確認する。

### version 3 objective修復・再実行承認

- ユーザーの「実行してください」を、推奨したGaussian soft-label structured likelihoodの実装と、同じcanonical kernelでのversion 3再実行承認として受領した。
- fixed exp209 decoder/state grid/transition、architecture、fold 0、seed 42、local true-state CE weight 0.25は変更しない。
- structured lossはhard quantized truth pathを廃止し、`log Z(unary) - log Z(unary + Gaussian label emission)`をtoken平均する。label observationはstate stepと同じ`0.35 ft`に固定し、sigma/loss gridを行わない。
- custom gradientは通常posteriorからlabel-conditioned posteriorを引く。疎なtruth jump自体を単一grammar-feasible pathに要求しないため、version 2のdata-contract failureを解消する。
- 再実行規模は`1 architecture x fold 0 x seed 42 = 1 neural model`、LightGBM config/booster `0/0`、PF/Beam 0、parent/control再学習0。Stage B、inference、submissionは実行しない。
- source SHA256: `f45c8cb772677f35cbd580180424a59bd4c91107e5d8910b1fe9ddc64fbb3e91`。canonical train Notebook SHA256: `dc9a6318ad410de260cc057b61f8a97296d38f65b9a83fde74acc83c51342298`。
- Jupytext `--test`、py_compile、Ruff、専用pytest `9 passed, 1 skipped`、strict experiment/template validationを通過した。skipはローカル環境にPyTorchがないためで、custom Torch posterior/gradientはKaggle T4が初回authoritative実行となる。
- repository全pytestは`342 passed, 1 skipped, 2 failed`。2件は未変更のexp296で、実行完了後config status/run flagと旧test期待値が不一致な既存状態であり、exp295専用testとは独立のためexp296を変更しない。
- Kaggle package notebook SHA256: `d36ff22a22257e89c32d38c6eb89d7dce94deeddfa50fe957073dba058f4c3eb`。bootstrap内configはloose/packageとbyte一致し、SHA256 `0f97ff57f617fbca22edc7493bc7a3139ae119d8f701b6905a5d0a1e3e56086e`、T4、internet off、run-on-push、exp209 kernel sourceを確認した。
- push前pullでcanonical id `kentookumura/exp295-prefix-gr-ssm-fold0-train`、id_no `127896241`の存在を再確認した。通常pullは確定済みversion 1 metadata（GPU off）を返すため、version 3 pushではCLIにも`--accelerator NvidiaTeslaT4`を明示する。
- `kaggle kernels push ... --accelerator NvidiaTeslaT4`で`Kernel version 3 successfully pushed`を確認した。run URLは`https://www.kaggle.com/code/kentookumura/exp295-prefix-gr-ssm-fold0-train`。
- Kaggle listの開始時刻は`2026-07-20T01:35:47.267Z`、初期statusは`KernelWorkerStatus.RUNNING`。running neural model / 学習完了 neural modelは`1 / 0`。
- ユーザーは枠解放後に連絡し、継続監視は止めてよいと指定済みのため、開始確認後の常駐pollingは行わない。完了連絡後に同一kernelの通常logsから結果を回収する。

### version 3 timeout診断

- ユーザーからtimeout連絡後、final status `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`を確認した。開始`2026-07-20T01:35:47.267Z`、診断時`2026-07-20T13:46:57Z`で約12時間経過しており、Kaggle runtime timeoutと整合する。
- 通常logsにtraceback、CUDA OOM、data path、network errorはない。最後の出力は18.455680111秒のscientific/cost contract previewで、epoch 1 summary、valid freeze、model/checkpoint、metricsは1件も出ていない。
- `kaggle kernels files`は`[]`。完了epoch/model/prediction/生成物/提出はすべて0。
- 分類: `kaggle_runtime_timeout_compute_infeasible`。version 2のhard truth path不整合は解消したが、Gaussian soft-label structured lossは各viewで通常posteriorとlabel-conditioned posteriorをそれぞれforward-backwardするため、exact DPが合計4 sweep必要になる。
- 実データshape auditはfit wells 556、fit views 1,668、suffix rows 8,571,405、平均suffix 5,138.732 rows/view、平均TVT grid 571.697 states、rate 41 states。soft-label DPだけで1 epoch `803,449,626,924` position-rate cells、最大8 epochs `6,427,597,015,392` cellsで、62 early-stop official views/epochとouter-valid 155 wellsの3 control decodeはこの外側に追加される。
- 1 epochも完了しなかったため、固定Stage A runtime gate `<=8.5 h`はFAIL。failure policyに従いexp295内のepoch/view/loss/sigma/band/architecture救済やversion 4再pushは行わず、Stage B、inference、submissionをbranch closeする。
- 次にGR whole-well unaryを再訪する場合は新しいexpが必要。候補は`local CEでunaryを学習しexact SSMは評価時だけ使う`か、`固定長windowだけでstructured trainingする`かで仮説が変わるため、ユーザー選択前に実装しない。

## 2026-07-19 Stage A Kaggle実行承認

### push前の固定計算量

- 対象: Stage A、fold 0、seed 42、active architecture 1。
- 今回実行するnew neural model: `1`（`1 architecture x 1 fold x 1 seed`）。
- LightGBM config / total booster: `0 / 0`。
- PF/Beam well-run: `0`。
- parent/control再学習: `0`。controlは同一trained modelのType Well GR circular shuffleとzero-unary readoutだけで、追加学習しない。
- exp209 baselineは保存済みcacheを参照し、再学習しない。
- GPU: Kaggle `NvidiaTeslaT4`、internet off、最大8 epochs、early stoppingあり。
- canonical kernel id: `kentookumura/exp295-prefix-gr-ssm-fold0-train`。
- Stage B fold 1-4、inference、submissionは承認範囲外で、Stage A結果を見ずに進めない。

### canonical採用

- `_compact_selfcontained_train.ipynb`を同一内容のcanonical train notebookへ採用する。
- canonical inference notebookは変更しない。

### 初回push受付エラー

- 初回候補`exp295 prefix anchored wholewell gr alignment ssm train`は55文字で、Kaggle `SaveKernel`がHTTP 400 `The title cannot exceed 50 characters.`を返した。
- 旧slug/titleではkernel/versionも実行も作成されていないことを自分のkernel一覧で確認した。
- 仮説・notebook・計算量を変えず、titleを45文字の`exp295 prefix anchored gr alignment ssm train`、canonical slugを`exp295-prefix-anchored-gr-alignment-ssm-train`へ短縮して再packageする。
- 短縮後の初回SaveKernelは`Maximum batch GPU session count of 2 reached`で実行開始前に拒否された。直近4 kernelはその後すべて`COMPLETE`と確認し、既存実行は停止していない。
- 同slugの再pushは`Notebook not found`、read-only確認はlistのrefなし、status 404、pull 500となり、Kaggle側に失敗した作成要求の予約状態が残ったと判断した。
- 旧slugに可視kernel/version/実行がないことを確認したうえで、仮説・notebook・計算量を変えず、未使用のcanonical slug/title `exp295-prefix-gr-ssm-stagea-train` / `exp295 prefix gr ssm stagea train`へ切り替える。旧予約状態は削除しない。
- その未使用slugもGPU create時点で同じsession上限に拒否された。空き確認目的のGPU create再試行でslug予約状態を増やさないため、最終canonicalを`exp295-prefix-gr-ssm-fold0-train` / `exp295 prefix gr ssm fold0 train`に固定する。
- 最終canonicalは同一notebookをprivate、GPU off、run offのdraftとして先に登録し、登録確認後にT4/run-on-push metadataで同slugを更新する。CPUでnotebook本体は実行しない。
- 最終canonical draft version 1の登録に成功した。Kaggle id_noは`127896241`、pullでprivate / GPU off / internet off / exp209 kernel sourceを確認した。version 1はrun offのためStage Aを実行していない。
- T4/run-on-pushへのversion 2更新は`Maximum batch GPU session count of 2 reached`で未作成。週次GPU quotaは`14.37h used / 30.63h remaining`で、quota不足ではなく同時session上限だけが待機理由。

## 2026-07-19 Stage A実装

### 承認範囲

- Stage Aの別名compact self-contained実装とcontract testsを承認として扱った。
- canonical Notebook採用、Kaggle package、GPU push、Stage B/C、inference、submissionは未承認のまま維持した。
- push時予定はactive architecture 1 x seed 42 x fold 0 = 1 neural model。LightGBM config 0、booster 0、PF/Beam well-run 0、control/parent再学習0。

### 実装内容

- mask-first horizontal loaderは`MD/X/Y/Z/GR/TVT_input`だけを読み、outer-valid `TVT`はglobal model/unary/posterior/control/row identity/SHA freeze後の別loaderで読む。
- exp202と同じsorted complete-well GroupKFold identity、outer-train stable early-stop holdout、official/256/512-row pseudo-cut manifestを実装した。
- finite median/MAD、missing mask、GR derivative、Type Well no-extrapolation、32-pair minimumのHuber affine summaryとmasked-attention prefix contextを実装した。
- embedding 64、dilation 1/2/4/8/16、GroupNorm、FiLM、bilinear cosine、clamped positive temperatureのcomplete-well unaryを実装した。
- exp209の`step=0.35`、41 rates、rate/position transition、start prior、bandを固定したlog-space exact forward-backward、posterior mean、marginal MAP、global Viterbi、std/entropy/edge massを実装した。
- structured CRF NLL + local CE、AdamW、AMP、gradient accumulation 4、outer-train early stoppingを実装した。
- 同一trained modelのreal、stable Type Well circular shuffle、zero-unary geometry-onlyをfreezeし、exp209保存済みcacheを再生成せず比較するStage A全条件AND guardを実装した。
- fail-closed inference候補はStage B promotionと別承認前のinference/submissionを拒否する。

### Notebook構成比較

- primary parent exp202にはcompact self-contained版がないため、通常Jupytext sourceを構成参照にした。
- exp202 train source 2234行に対し、exp295 compact train sourceは約2100行で、Imports、runtime/config、fold/input、preprocessing、neural emission、fixed decoder、training、freeze/readout、orchestrationをすべてNotebookセルで追える。
- canonical scaffoldは6 cellsのまま上書きせず、別名compact候補を生成した。

### 実装検証

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py> <compact_inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_inference.py>
.venv/bin/python -m py_compile <compact_train.py> <compact_inference.py>
.venv/bin/ruff check <compact_train.py> <compact_inference.py> experiments/exp295_prefix_anchored_wholewell_gr_alignment_ssm/tests/test_exp295_prefix_anchored_wholewell_gr_alignment_ssm.py --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp295_prefix_anchored_wholewell_gr_alignment_ssm/tests/test_exp295_prefix_anchored_wholewell_gr_alignment_ssm.py
.venv/bin/pytest -q
make validate-exp EXP=exp295_prefix_anchored_wholewell_gr_alignment_ssm
make validate-template
python3 .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp295 --root .
```

- Jupytext train/inference `--test`: PASS。
- py_compile / Ruff: PASS。
- 専用pytest: `9 passed, 1 skipped`。skipはローカル`.venv`にPyTorchがなく、exact Torch posterior/structured-gradient testだけを実行できないため。ロジックはKaggle GPU runtimeで初回実行する。
- repository pytest: `307 passed, 1 skipped`。
- strict experiment validation / project template validation: PASS。
- experiment-doc reviewer: core evidence categories present。`metrics.json`のbase/changes指摘へ`diff_summary`を追加した。
- `__file__`参照: 0。canonical scaffold上書き: 0。Kaggle package/push/run: 0。

## 2026-07-19 設計確定

### 作成コマンド

```bash
make new-steering EXP=exp295_prefix_anchored_wholewell_gr_alignment_ssm
make new-exp EXP=exp295_prefix_anchored_wholewell_gr_alignment_ssm
```

### 根拠

- exp178はreal GR pair AUC 0.7654、shuffled 0.6623でlearned matching signalを支持した。
- exp202はexisting + heatmap top10 oracle RMSE 2.7455でdistributed heatmap headroomを示した。
- exp197はlocal CNN scorerのreal-shuffled AUC差が+0.006だけで、短いlocal window中心設計を否定した。
- exp221はfixed exact HMMでexp148 8.5013から8.3277へ改善し、global decoderの価値を支持した。
- exp244のlocal-linear pseudo-start risk相関-0.018は、pseudo-startをconfidence gateにしない根拠とした。
- exp292はhand-crafted Type Well warp scorerのAUC lift -0.047、eligible 3.75%で閉鎖済み。learned whole-well emissionは同じscore/coverage救済ではなく別仮説として扱う。
- exp293 Stage 2/3は固定物理candidate bankの離散選択で、continuous state unaryを学ぶexp295とは分離した。

### 固定した設計

- 1 complete well = 1 sample。
- target-well horizontal GR / Type Well GR / known prefix / target-well trajectoryだけを入力にする。
- shared multi-scale 1D encoder、32次元prefix context、FiLM、bilinear cosine unary。
- exp209 state grid/transitionを固定し、known prefix hard clamp + exact forward-backward + posterior meanを使う。
- outer-train official/256/512-row pseudo-cut viewでstructured NLL + local CEを学習する。
- test-time gradient adaptation、neighbor well、candidate bank、hard top1、既存ML/PF blendは禁止する。
- Stage Aはfold 0の1 model、Stage Bはfold 0を再利用して追加4 models、Stage Cはpromotion PASS後だけ。
- LB 5.x promotionはpooled OOF 6.0以下、stretch 5.0以下。GR controls、5 folds、1000+、hidden-like、p95/worst guardを全必須とする。
- 6.0～6.75かつGR attribution PASSは別expのarchitecture iterationだけを許し、exp295 inferenceへ進めない。

### 設計検証

```bash
make validate-exp EXP=exp295_prefix_anchored_wholewell_gr_alignment_ssm
make validate-template
make update-summary
python3 .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp295 --root .
```

- strict experiment validation: PASS。
- project template validation: PASS。
- `config.yaml` YAML parse / `metrics.json` JSON parse: PASS。
- steeringと実験設計文書の未記入placeholder: 0。
- automated experiment-doc review: core evidence categories present。
- raw schema確認: horizontal file columnsは`MD/X/Y/Z/.../TVT/GR/TVT_input`、Type Well globは`*__typewell.csv`。model input allowlistはhorizontal fileの`MD/X/Y/Z/GR/TVT_input`へ固定し、well/idはfilename/canonical row identityから導出する設計へ修正した。
- `experiment_summary.md`へexp295のlineage/status/descriptionを追加した。

## 変更点

- `KAGGLE_DIRECTION.md`へ最優先・高リスク・stage-gated exp295を追加した。
- 旧`heatmap_unary_exact_hmm_redecode_probe`は、exp295のlineage/evidenceへ統合して独立backlogから削除する設計にした。
- steering 3文書、`architecture_contract.md`、`config.yaml`、README、SESSION_NOTES、result、metricsへ同じ固定契約を反映した。
- 設計確定時点ではscaffold train/inference Notebook、`settings.py`を変更せず、実験コード、Jupytext source、tests、package、Kaggle run、output、prediction、submissionを作成しなかった。
- その後のStage A実装承認で別名Jupytext source/notebookとtestsだけを追加した。canonical scaffold、settings、package、Kaggle run、output、prediction、submissionは未変更・未作成である。

## 再現性メモ

- seed policy: fixed global 42 + stable SHA256 per well/fold/pseudo-cut/control。
- stochastic components: 将来のPyTorch CUDA/AdamW/dropout/seeded dataloader。
- CPU/GPU runtime: 現在は未実行。Stage AはT4、1 well/batch、DataLoader worker 0、AMP予定。
- deterministic anchor: false。CUDA/AMP byte parityを仮定しない。
- Kaggle kernel id / version: なし。
- input / fold / pseudo-cut / feature SHA: 未生成。実行時の必須記録としてconfigに固定済み。
- model / emission / posterior / prediction SHA: 未生成。
- submission SHA: 対象外。
- rerun check: 未実行。

## 次のアクション

1. canonical train NotebookとKaggle packageを生成・検証する。
2. fold 0の1 neural modelだけをKaggle T4で実行し、logs/statusを監視する。
3. Stage A metricsとartifact SHAを記録する。
4. Stage A全PASSまでfold 1-4、inference、submissionへ進まない。
