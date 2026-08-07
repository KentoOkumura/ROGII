# exp282_longtail_prediction_zone_self_gr_loop_closure_readout セッションノート

## 目的

prediction-zone内のearlier GR motifをlong-tail receiverのtarget-free pseudo-anchorとして使えるか、
補正前にloop-closure edge精度とexp263 donor-transfer方向だけを監査する。known `TVT_input` prefixを
matching donorに使わず、exp281の完了にも依存しない独立0-booster readoutとして設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 1完了、technical PASS / scientific FAIL、branch closed
- CV: prediction CVなし（train-side readout）
- LB: 対象外（inference / submission disabled）
- active variant / LightGBM config / trained fold / booster: `1 / 0 / 0 / 0`
- HMM / PF variant: `0 / 0`
- parent/control再学習: 0
- GPU / inference / submission: off / disabled / disabled
- Kaggle push approval: true（2026-07-19 ユーザー「実行してください」）

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
make new-steering EXP=exp282_longtail_prediction_zone_self_gr_loop_closure_readout
make new-exp EXP=exp282_longtail_prediction_zone_self_gr_loop_closure_readout
```

初期template作成時点ではKaggle prepare / push、notebook実行、学習、推論、提出は実行していなかった。

2026-07-19の実装依頼後、次を実行した。

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/exp282_longtail_prediction_zone_self_gr_loop_closure_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/exp282_longtail_prediction_zone_self_gr_loop_closure_readout_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/exp282_longtail_prediction_zone_self_gr_loop_closure_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/exp282_longtail_prediction_zone_self_gr_loop_closure_readout_compact_selfcontained_inference.py
.venv/bin/python -m py_compile experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/*compact_selfcontained*.py
.venv/bin/ruff check experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/*compact_selfcontained*.py tests/test_exp282_longtail_prediction_zone_self_gr_loop_closure_readout.py
.venv/bin/pytest -q tests/test_exp280_exp226_shift_likelihood_separability_readout.py tests/test_exp281_exp226_residual_offset_exact_hmm_transition_probe.py tests/test_exp282_longtail_prediction_zone_self_gr_loop_closure_readout.py
make validate-exp EXP=exp282_longtail_prediction_zone_self_gr_loop_closure_readout
make validate-template
make test
```

検証結果:

- exp282 synthetic tests: 6 passed
- exp280/281/282 targeted tests: 18 passed
- ruff / py_compile / Jupytext train+inference round-trip: PASS
- strict validate-exp / validate-template: PASS
- repository tests: 178 passed / 1 failed
- 全体testの1件FAILは今回未変更のexp264で、configのinference status
  `corrected_inference_v4_complete`とtest期待値`user_authorized_2026_07_19`の既存不一致だった。

## 固定scientific contract

- 親: `exp090_lateral_self_gr_match_pseudotail_probe`
- 固定予測参照: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 並行比較: `exp281_exp226_residual_offset_exact_hmm_transition_probe`。結果待ちは不要。
- donor: same-well prediction zoneの`0 <= md_since < 500 ft`
- receiver: same-well prediction zoneの`md_since >= 1000 ft`
- GR: interpolation、rolling mean 5、window z-normalize
- match: half-window `[8,15,25]`、primary 25、stride 3、forward/reverse NCC
- confidence: NCC percentile、best-second gap percentile、multiscale agreement、segment supportの等重み
- negative control: real edge freeze後のwell内donor assignment shuffle
- truth attachment: real/shuffled edge content SHA保存後のみ
- correction / model / HMM / PF / inference / submission: 全て対象外

## 実装境界

- `*_compact_selfcontained_train.py` / `.ipynb`を別名で作成した。train sourceは1,627行、10章で、
  類似0-booster readoutのexp280 train source 1,165行に対してinput/matching/freeze/readout/orchestrationを
  欠かさず展開している。
- 同じexp内helper importは使わず、notebook unsafeな`__file__`参照もない。
- score-stageはraw horizontalの`MD`、`GR`、`TVT_input`とexp263 Stage 0 identityだけを読む。
- real/shuffled edgeを保存・logical content SHA固定後にだけraw true TVT、exp263 fixed OOF、fold、
  hidden-like roleをattachする。
- rolling mean 5、half-window 8/15/25、stride 3、forward/reverse NCC、固定tie-break、gap、
  multiscale agreement、segment support、equal-weight confidence、well-local SHA256 shuffleを実装した。
- overall / fold / distance / hidden-like / orientation / by-well / donor-transfer metricsとtechnical/scientific
  guard、edge/schema/input/output SHA保存を実装した。
- `*_compact_selfcontained_inference.py` / `.ipynb`はfail-closedである。
- ユーザーの2026-07-19実行依頼を明示採用承認として、compact sourceから正規train/inference
  `.ipynb`を生成した。
- `settings.py`は正規stub notebook用templateのままだが、compact sourceはself-containedで参照しない。

## 再現性メモ

- seed policy: real edgeはRNGなし。shuffled controlだけstable SHA256 per-well local RNGを使う。
- stochastic components: within-well shuffled donor negative controlのみ。
- parallel RNG: global RNG / Python hashを禁止し、wellごとのlocal RNGへ分離する。
- process: sorted wells、well逐次、receiver chunk 256、full pairwise matrix保存禁止。
- CPU/GPU runtime: Kaggle CPU / GPU offでversion 1を248.206秒で完了した。
- input / edge schema / edge logical content SHA: Kaggle outputで生成し、取得ファイルとlogのSHA一致を確認した。
- model manifest / model SHA: fitted modelなし。frozen edge content SHAでreadout境界を固定した。
- prediction / submission SHA: 補正・submissionを作らないため対象外。
- deterministic anchor: 初回成功だけでは宣言しない。

## 次のアクション

固定scientific guardがFAILしたため、parameter rescue、補正、HMM/PF接続、inference、submissionへ進まない。
新規救済backlogを追加せず、loop-closure branchを完了negativeとして閉じる。

## 2026-07-19 Kaggle CPU v1 実行承認

ユーザーの「実行してください」を、別名compact notebookの正規notebook採用と、固定契約による
Kaggle CPU train-side readout 1回の実行承認として記録する。

- active audit variant: 1
- LightGBM / model config: 0
- trained fold: 0
- total booster: 0
- HMM variant / well-run: 0 / 0
- PF variant / well-run: 0 / 0
- parent/control再学習・再生成: 0
- runtime: Kaggle CPU / GPU off / internet off / single process
- input kernel sources: exp263 Stage 0 v1、exp115 hidden-like assignment
- correction / inference / submission: disabled / disabled / disabled

credential preflightはAPI tokenが未設定、OAuth credentialとlegacy credentialはOK。Kaggle CLIはOAuthを
利用できるため続行する。正規notebook採用、strict validation、package metadata/bootstrap SHA監査後に、
canonical kernel idへ1回だけpushする。

### 正規notebook採用・package監査

compact train/inference sourceから正規`*_train.ipynb` / `*_inference.ipynb`を生成し、template stubを
置き換えた。正規trainは22 cells / 87,093 bytes / SHA
`f754b3cecc8928dd41db93572f9e85bfb6797e1f42fce07874be98fb9c401b7b`、inferenceは8 cells /
4,692 bytes / SHA `13ef4e128e4b69960ce814f259345e44a1f56a975c3f2ef9854a1e0fbf8ac683`。

```bash
make prepare-kaggle-notebooks \
  EXP=exp282_longtail_prediction_zone_self_gr_loop_closure_readout \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp282-longtail-self-gr-loop-closure-readout-train --title 'exp282 longtail self gr loop closure readout train' --run-on-push --strict"
```

package監査:

- kernel id/title slug: `kentookumura/exp282-longtail-self-gr-loop-closure-readout-train` /
  `exp282 longtail self gr loop closure readout train`、一致
- private / CPU / GPU off / TPU off / internet off / run-on-push true
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources: exp263 Stage 0 train、exp115 hidden-like train
- bootstrap 15 files、manifest/hash/bytes/loose file mismatch 0
- config SHA: `1895a80322c81cdd0fea19bdca1030903e93825fa4e1d6f26a89804d45aea684`
- train source SHA: `9834212761f2c415346904dcd761067dd38446645088bb58c8a16b3639a434a9`
- inference source SHA: `34f5f5846be6ef70e9dc510101b290979a8cd3424aa38a67d07d8b5a9f3614a9`
- metadata SHA: `46f22ad33b3d2d46d7c1384c9e7d840f7f84c7ff72fb557edb92cef64ae92d42`
- packaged notebook SHA: `eb2b8faebbde07944eaedd3369a649fc55738e78cff8dfc91852819615762243`
- exp282 tests 6/6、strict validate-exp PASS

### Kaggle CPU version 1 push

```bash
make push-kaggle-train EXP=exp282_longtail_prediction_zone_self_gr_loop_closure_readout
kaggle kernels pull kentookumura/exp282-longtail-self-gr-loop-closure-readout-train \
  -p /tmp/kaggle-pull-exp282-longtail-selfgr-v1 -m
kaggle kernels logs kentookumura/exp282-longtail-self-gr-loop-closure-readout-train
```

- push: success、version 1
- kernel: `kentookumura/exp282-longtail-self-gr-loop-closure-readout-train`
- id_no: `127838798`
- URL: `https://www.kaggle.com/code/kentookumura/exp282-longtail-self-gr-loop-closure-readout-train`
- pull metadata: private、CPU (`machine_shape=None`)、GPU/TPU/internet off、competition source 1、
  kernel sources 2でlocal packageと一致
- 初回logs: 空。Kaggle CLIの実行中log非表示仕様として扱い、同じkernel idの完了logを後段で取得した。

## Kaggle CPU version 1 完了結果

同じkernel idを監視し、Kaggle API status `COMPLETE`と最終logを確認した。全773 wellsを処理し、
notebook runtimeは248.206秒だった。実行契約は1 audit variant / LightGBM config 0 / trained fold 0 /
booster 0 / HMM/PF well-run 0のまま完了した。

### Technical guard

- canonical rows / wells: 3,783,989 / 773
- eligible receiver center / generated edge: 997,733 / 997,733、coverage 1.0
- finite score coverage: 1.0
- expected folds: `[0,1,2,3,4]`
- forbidden score-stage column: 0
- truth attachment before edge freeze: 0
- technical guard: PASS

### Scientific readout

- all-edge within10: 0.550644 vs shuffled 0.549871、lift +0.000774
- high-confidence within10: 0.554309 vs shuffled 0.551052、lift +0.003257
- high-confidence within10 lift by fold:
  `[-0.000399, +0.006493, +0.005594, +0.001749, +0.002847]`、positive 4/5
- all-edge positive lift: 3/5 folds
- high-confidence median delta改善: 4/5 folds
- high-confidence receiver coverage: 3.323%、PASS
- hidden-like spatial / typewell-purged high-confidence lift: +0.005491 / +0.004300、両方PASS
- high-confidence donor-transfer: baseline 8.954770 ft、matched 15.849509 ft、gain -6.894739 ft
- donor-transfer fold gain:
  `[-6.469817, -6.432448, -6.422177, -7.317443, -7.775890] ft`、改善0/5
- scientific guard: FAIL
- final decision: `close_branch_without_parameter_rescue`

### 成果物確認

AGENTS.mdの方針に従い、まずKaggle logsを評価した。fold数値、by-well transfer、成果物SHAの実ファイル
確認が必要だったため、その後`kaggle kernels output`で85 MBのoutputを`/tmp`へ取得した。大きな
target-free edgeや生artifactはrepositoryへ保存していない。

- frozen edge logical content SHA:
  `2b9ecbb956e2b84ee61ddefeb54ed0fcca98b984e76ba1be0e7c9321f5f74c28`
- edge gzip raw / decompressed SHA:
  `fa70053b4f290e2bb487bca2b48e99389b9db0f57e43b3df448c9b05b9e9d297` /
  `e2a425bde45ef8abea59838cf734856d6c5c27671503ce70305320ced9a408a1`
- summary SHA: `32896b76ad069fb7bc569ce8ab1c6b6e389d0f06800f5d7ef2c0cc27d38894e9`
- by-well / donor-transfer / contract / schema / fold / input / orientation / overall / scope /
  well-manifestの10 file SHAもlog値と取得ファイルで全一致した。

### 解釈

self-GR confidenceはshuffled比で弱いpooled signalを持つが、fold-stableな5/5方向性も60% precisionもない。
matched donorのexp263 fixed予測transferは全foldで6.42～7.78 ft悪化し、GR motif一致をabsolute TVT
pseudo-anchorとして使用できない。事前契約どおりthreshold/window/stride/confidence/donor範囲の救済、
soft correction、raw-test inference、submissionは行わない。
