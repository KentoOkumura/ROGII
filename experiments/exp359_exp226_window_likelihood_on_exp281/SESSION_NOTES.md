# exp359 セッションノート

## 目的

旧exp325のwindow-likelihood仮説をexp323/338 chainから分離し、
exp281直結の0-HMM Stage 0として固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0完了・固定gate FAIL・救済なし閉鎖
- CV: window MRR `0.372904` / saved control `0.395168`
- LB: なし

## 2026-07-23 設計

- ユーザー依頼によりexp359を採番し、steeringとscaffoldを作成した。
- parentはexp281、Stage 0 saved Gaussian controlはexp280に固定した。
- Stage 0はscientific score 1 / saved control 1 / 5 reporting folds /
  HMM・model・trained fold・booster各0。
- Stage 1予約は1 variant / 773 HMM runs / parent-control再実行0。
- 実装、Notebook採用、Kaggle package/push/run、inference、submissionは行っていない。

## 2026-07-25 Stage 0実装

- ユーザーの「exp359を実装してください」をStage 0実装承認として受領した。
- 正規`*_train.ipynb` / `*_inference.ipynb`は既存placeholderのまま保持し、
  compact self-contained Jupytext train候補とfail-closed inference候補を別名で追加した。
- full 500-row / stride 125 / 13 shiftsのwindow scoreを実装した。
- scoreはexp226 sourceに合わせてknown-prefix GRをType Well GRへaffine calibrationし、
  correlation `2.0`、MSE `0.5`、level `0.1`の固定式で計算する。
- 13 shift raw scoreをstate方向へ標準化し、softmax posterior SDから
  `125/500 * clip(1.1-0.12*sd,0.3,1.0)`のlambdaを計算する。
- negative controlはwell/window/profile SHA由来のlocal RNG permutationとし、
  global RNGや並列順序に依存しない。
- saved exp280 controlはdecompressed/content/scientific-contract SHAをhard guardし、
  各window centerの`suffix_offset // 512`に対応する保存block scoreをload-onlyで使う。
  部分block scoreの推定やcontrol再生成は行わない。
- window identity、profile SHA、eligibility、raw/normalized/potential score、
  posterior SD、lambda、shuffle、control mappingをtruth前にbundle化してcontent SHAを
  固定する。exp226 OOFの`tvt_true`はその後だけ読む。
- Stage 0 gateはpooled MRR/top3各`+0.01`、各4/5 folds、
  real>shuffle 5/5 folds、1000+・hidden-like 2面正方向、
  eligible window `>=0.25`のANDを維持した。

実行量契約:

- Stage 0 scientific score / saved control: `1 / 1`
- shift candidates / reporting folds: `13 / 5`
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- parent/control再実行: 0
- Kaggle package/push/run、inference、submission: 0
- Stage 1予約: Stage 0全gate PASSと別承認時だけ1 variant / 773 HMM well-runs

実装・検証:

```bash
.venv/bin/python -m py_compile \
  experiments/exp359_exp226_window_likelihood_on_exp281/*compact_selfcontained*.py \
  experiments/exp359_exp226_window_likelihood_on_exp281/tests/test_exp359_exp226_window_likelihood_on_exp281.py
.venv/bin/ruff check \
  experiments/exp359_exp226_window_likelihood_on_exp281/*compact_selfcontained*.py \
  experiments/exp359_exp226_window_likelihood_on_exp281/tests/test_exp359_exp226_window_likelihood_on_exp281.py \
  --select F821,F401,F841,E722,E501
.venv/bin/pytest -q \
  experiments/exp359_exp226_window_likelihood_on_exp281/tests/test_exp359_exp226_window_likelihood_on_exp281.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 \
  experiments/exp359_exp226_window_likelihood_on_exp281/*compact_selfcontained*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp359_exp226_window_likelihood_on_exp281/*compact_selfcontained*.py
make validate-exp EXP=exp359_exp226_window_likelihood_on_exp281
```

- 専用test: `8 passed`
- exp280/exp357/exp359/Notebook関連test: `28 passed`
- py_compile / Ruff / Jupytext train・inference / strict validation: PASS
- compact train notebook: 21 cells（code 10 / markdown 11）、87,537 bytes
- compact inference notebook: 10 cells（code 5 / markdown 5）、6,394 bytes
- `__file__`、同一exp helper import、`from src`依存: 0
- config SHA:
  `382dddc272bdbafbfb5ce3c96481364cc92cf962656b10c4f0198b90b50fa158`
- train source / notebook SHA:
  `cd17838b825018b0143585415a7ad69000a519c601e66e732b399d3ba93a132b` /
  `ed2da68690977bbb18e9d5bb52a864614eafe2f061f1b638a33d8024215c2c74`
- inference source / notebook SHA:
  `dd327c05734e694e1c882d74286d486cd1e83a88c6d0aed1289b725c298fc484` /
  `ac3b977f8ff34844302675a7b8cdf1b9131dc48842454e9fa99c2b7a307eef2f`
- test SHA:
  `8e2162af94b4528190c4de656498be3eeb5b80ac0ff6304f199240d4c339dae4`

親/参照構成比較:

- exp281正規self-contained trainは1,526行/10章でfull exact-HMM生成を扱う。
- exp280正規self-contained trainは1,165行/9章で512-row Gaussian shift readoutを扱う。
- exp359 compact trainは1,639行/9章でruntime/input/window scoring/control alignment/
  truth-late readout/gate/orchestrationを展開する。Stage 1 HMMは意図的に含めない。

リポジトリ全体test:

```bash
make test
```

- `967 passed, 6 skipped, 5 failed`
- exp359専用8件は全体実行でもPASS。
- 失敗2件は既存exp296の完了状態・実行flagと旧test期待の不一致。
- 残る3件は既存exp348、exp358、exp391の承認/active statusと旧test期待の不一致。
- 上記対象のconfig/testは今回変更しておらず、exp359実装のためには修正しない。

## 再現性メモ

- real scoreはRNGなし。negative controlはwell/window content SHA由来の固定permutationを使う。
- exp226 OOF、exp280 score/contract、exp281 predictionのSHAをhard guardする。
- window manifest、score bank、eligibility、lambda、control、readoutのcontent SHAを記録する。
- Stage 1時だけdecoder contractとprediction SHAを記録する。
- deterministic anchorとは扱わない。

## 実装時点の次のアクション（完了）

正規train Notebook採用とKaggle private CPU Stage 0実行は別承認とする。
Stage 0 PASSでも773 HMM runsへ自動進行しない。

## 2026-07-25 Kaggle Stage 0実行承認

- ユーザーの「実行してください」を、compact self-contained train候補の正規Notebook採用、
  Kaggle package/push、private CPU Stage 0を1回実行する承認として受領した。
- scientific score / saved control: `1 / 1`
- shift candidates / reporting folds: `13 / 5`
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- parent/control再学習・再生成: 0（exp280 controlはSHA固定load-only）
- runtime: Kaggle CPU、GPU/TPU/internet off
- canonical kernel:
  `kentookumura/exp359-exp226-window-likelihood-on-exp281-train`
- canonical title: `exp359 exp226 window likelihood on exp281 train`
- 未承認: Stage 1の1 variant / 773 HMM runs、inference、submission
- credential checkerはOAuth/legacy CLI credentialを確認した。API Tokenは未設定だが、
  既知private kernelのpullに成功し、CLI OAuth認証が有効であることを確認した。
- canonical slugの事前pullは403、既知private kernelのpullは成功したため、
  exp359は未作成の新規canonical kernelとして扱う。

正規採用・事前検証:

- 正規train notebookはcompact候補とcell source `21 / 21`一致。
- 正規train notebook: 21 cells（code 10 / markdown 11）、
  SHA `ed2da68690977bbb18e9d5bb52a864614eafe2f061f1b638a33d8024215c2c74`
- 承認済みconfig SHA:
  `f24d4e47ce8c0e72584235a77263e1b84ec9ff5a3e13b93bd1c276822c951a32`
- compact train source SHA:
  `cd17838b825018b0143585415a7ad69000a519c601e66e732b399d3ba93a132b`
- 専用＋Notebook tests: `12 passed`
- py_compile / Ruff / Jupytext round-trip / strict experiment・project validation: PASS

Kaggle package:

- metadata: private、CPU、GPU/TPU/internet off、run-on-push
- competition source: 1、kernel sources: exp226 / exp280 / exp115の3件
- package notebook: 22 cells（bootstrap code 1 + 正規21）、
  SHA `365a829e3de90754c3c792f8c268500b8906a9e9eb26beb502da06478c1ef35f`
- metadata SHA:
  `d1d63c8b25301c1d1b03c3f3e98da42c04c10504efa79bf06f42d20e6075192c`
- config loose/package/bootstrap SHA:
  `f24d4e47ce8c0e72584235a77263e1b84ec9ff5a3e13b93bd1c276822c951a32`
  で一致。
- bootstrap support files: 5、ZIP SHA:
  `bd1a63f51a90d3018098f0a6b6840116146f696f14fa172061fd014c32346d3e`
- bootstrap内train source SHA:
  `cd17838b825018b0143585415a7ad69000a519c601e66e732b399d3ba93a132b`
  で正規sourceと一致。

push・canonical照合:

- canonical private CPU kernel version 1をpushした。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp359-exp226-window-likelihood-on-exp281-train`
- canonical id_no: `128528648`
- pull metadata: id/title、private、CPU、GPU/TPU/internet off、
  competition source 1、kernel sources 3がpackageと一致。
- remote notebook: 22 cells、local packageと正規化後cell source `22 / 22`一致。
- 初期status: `KernelWorkerStatus.RUNNING`
- 重複実行防止のため、push直後にroot configの`run_stage_0=false`、
  `train_run_on_push=false`へ戻した。Kaggleへ送信済みversion 1のbootstrap configは
  SHA `f24d4e47...951a32`の承認済みtrueを保持する。

## 2026-07-25 Kaggle Stage 0結果

- canonical private CPU version 1（id_no `128528648`）は
  `KernelWorkerStatus.COMPLETE`。
- generated at: `2026-07-25T02:00:47.919160+00:00`
- runtime: `4523.211266517639 sec`
- rows / wells: `3,783,989 / 773`
- scientific score / saved control / shifts / reporting folds:
  `1 / 1 / 13 / 5`
- HMM well-run / model config / trained fold / booster:
  `0 / 0 / 0 / 0`
- parent/control再実行、Stage 1、inference、submission: 0

技術結果:

- candidate / eligible windows: `27,561 / 10,628`
- eligible fraction: `0.38561735786074525`
- eligible wells: `729 / 773`
- aligned saved-control blocks: `7,326`
- score finite / row identity / saved-control rank parity /
  quantization coverage: `1.0 / 1.0 / 1.0 / 1.0`
- real > shuffle: `5 / 5 folds`
- mean posterior SD: `26.5115837579042 ft`
- lambda: mean/min/max `0.075 / 0.075 / 0.075`、unique 1

科学結果:

- pooled window / control MRR:
  `0.37290353083540895 / 0.3951676655736686`
- pooled MRR差: `-0.022264134738259667`
- pooled window / control top3:
  `0.4144712081294693 / 0.44796763266842304`
- pooled top3差: `-0.03349642453895374`
- window - shuffle MRR / top3:
  `+0.12463593128111386 / +0.18046669175762137`
- MRR / top3改善fold: `0 / 5`, `0 / 5`
- fold別MRR差:
  `[-0.014113, -0.025702, -0.005178, -0.031855, -0.030024]`
- fold別top3差:
  `[-0.014743, -0.043925, -0.014001, -0.047371, -0.041473]`
- long-tail 1000+ MRR/top3差:
  `-0.015922152104981047 / -0.02483014206300188`
- hidden-like spatial MRR/top3差:
  `-0.018404308583851614 / -0.019949220166848025`
- hidden-like typewell-purged MRR/top3差:
  `-0.017773923002681158 / -0.015250544662309351`
- fixed guard: `stage_0_failed_close_without_rescue`
- Stage 1 eligible: false

生成物・SHA:

- scientific contract:
  `43aa952498f2fd1474bcca8c7bf651a854b2fc6f671384936ab67c795afd671a`
- target-free score content:
  `8a4c5623c5836a734dcd6bd44ff5b214546afaa7316ae1fdb882f0ce6f344c4a`
- window readout content:
  `617cfd7ee7239f6df71a6effbf50f8fc1fcf10943ffed495385aef81520dd1d9`
- gate / fold metrics / scope metrics:
  `956d194661828ab9766f9d632288130579c39df2e9b1fe8d3cb287bd9b1a9d05` /
  `5f3eaa7aee9bab139a21988701a56985a73171647e4520e151902fb2ad747f2e` /
  `4f29d2e6c61835767fd9957523b1df52449b30a0f1e81df024b7dbaedb412b30`

logsは完了後に取得した。logsにはfold/scope個別値がなかったため、
実ファイル確認が必要な根拠を満たすものとしてKaggle outputを一時ディレクトリ
`/tmp/exp359-output-v1.MrBnCj`へ取得し、gate/fold/scope metricsとSHAを確認した。
リポジトリの`artifacts/`へ大きな生成物はコピーしていない。

判定:

- 技術gateとnegative controlは成立したが、window scoreはsaved Gaussian controlより
  pooled、全5 folds、全3 stress scopeで悪い。
- lambdaは全eligible windowsで下限へ飽和した。ただし正の定数lambdaはwindow内の
  shift rankを変えないため、主因は固定500-row profile score自体のcontrol比不足。
- 同じOOFを見たwindow/stride/weight/lambda gridは事前禁止した救済になる。
- Stage 1の773 HMM runs、inference、submissionへ進まず、exp359を閉じる。

## 次のアクション

同じ500-row window-likelihood familyの救済実験は追加しない。
関連するregistration-offset候補はexp359の負結果を継承し、
full unknown-suffix window scoreを再利用せず、独立したknown-prefix
rolling-origin gateを通す場合だけ検討する。
