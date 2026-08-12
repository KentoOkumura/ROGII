# 設計

## 仮説

hjyact version 2の完成版final predictionは、exp510で採用した別公開sourceのpre-override成分より
exp413のPublic誤差を補完する。source finalを欠落なく再現して等率で混ぜれば、exp413単独および
exp510の公開3桁`7.201`を改善できる。

## アプローチ

exp510はhjyact/Leonid系の完成版ではなく、別公開sourceのpre-override成分をexp413へ10%混ぜたため、
Public LBはexp413と同じ`7.201`だった。本実験ではhjyact version 2が実際に採用したfinal stackを
そのままdynamic replayし、ユーザー指定の`0.50 / 0.50`式でexp413と等率blendする。
実行時は両経路を順番に丸ごと再実行せず、定義が同一のdeterministic candidate/feature blockを単一DAGで
先に1回だけ生成し、hjyact learned trajectoryとexp413へfan-outする。

**Assumption:** hjyact version 2のPublic LB `6.568`が同じPublic scoring rows上で再現される場合、
RMSEの三角不等式から固定50% blendは
`0.50 * 7.201 + 0.50 * 6.568 = 6.8845`以下となる。この値は期待値やCVではなく、両成分の公開scoreが
正確に再現された場合だけ成立する上限である。

## 実験範囲

- 対象実験: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- Route: `ensemble`
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- 比較対象: `exp510_exp413_exact_public_preoverride_hedge`
- 変更する変数: public成分をhjyact version 2 finalへ置換し、最終weightをユーザー指定`0.50 / 0.50`にする。
- 固定する変数: exp413、hjyact source path、float64演算、dynamic sample契約、提出形式。
- 新規学習量: scientific variant 1、LightGBM train config 0、新規booster 0、親/control再学習0。
  source parityに必要なRidgeは1 config × 5 folds。保存modelの正確な読込数は実装preflightでmanifest化し、
  push前に確定する。

## 公開source契約

- author / slug: `hjyact/ultimate-pf-config-strategy-a-reproducible-score`
- kernel id: `128161011`
- version / run: `2` / `337064157`
- source Public LB: `6.568`
- source snapshot SHA256: `cced2c1a85b7afd94368e7dd128a29cde1a7a577147896c94f8d2d8e412fc380`
- visible final CSV SHA256: `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`
- Kaggle image digest: `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`
- original runtime: GPU、約787.8秒、14,151行・3 well。

sourceの有効レイヤーは次の順序で固定する。

1. SP45/PF anchorを生成する。
2. 保存modelを使うlearned trajectoryを生成する。
3. `0.60 * SP45 + 0.40 * learned`を作る。
4. `RUN_GUARDED_OVERLAP_OVERRIDE=True`をsource閾値のまま適用する。
5. visible-prefix calibrationを`profile=balanced`、`final_selection=profile`、cut fractions
   `(0.50, 0.65, 0.75)`、cal/final seeds `24/48`、particles `350`で適用する。
6. model-package correctionはユーザー指定のruntime短縮として無効化し、model-package datasetをruntime入力から外す。
   version 6ではp95 guardで最終weightが0だったため、visible final値は不変となる見込みだが、既知SHAで再確認する。
7. PF seed-branch hedgeをstrength `0.60`、minor mass `0.25`、separation `[4,40]`、cap `2 ft`で
   適用し、これを`hjyact_v2_final`とする。
8. exp413のdynamic predictionをCSV boundaryで読み戻し、
   `final = 0.50 * exp413 + 0.50 * hjyact_v2_final`をfloat64で一度だけ書く。

現在のvisible sampleでguarded overlapが3 wellへ適用、balanced overlayの追加moveが0、version 6のmodel-packageが
p95 guardで最終weight 0、branch hedgeが1 wellへ適用されたことはsource parity用の署名であり、hidden sampleへ
固定する件数条件ではない。

## 2026-08-05 runtime短縮（ユーザー承認済み）

- SP45はtest wellごとの128-seed likelihood PF、beam、selectorを1 taskとして、joblib threadを最大4本使う。
  各taskはseed `0..127`を従来順で消費し、親processはtest-well入力順でrow/reportを連結する。
- exp413 exact HMM / self-GR HMMはwell taskをjoblib thread最大4本で実行し、結果をwell入力順で連結する。
- exp413 route-specific PF (`pf_ancc` / `pf_z`)もwell taskをthread最大4本で実行し、共有frameの更新は
  task内で行わず、親processが返却値を入力順に書き戻す。
- exp413 K16はHaswell subprocess内でwell taskをthread最大4本にし、train-only fieldとpinned kappaをread-only共有する。
- `n_jobs=min(4, test_well_count)`とし、0 wellは従来どおりfail-closeする。既知visible component SHAまたは
  既承認numerical witness以外は受け入れない。
- version 7のvisible 3 well実測では全体`1,197.667秒`、v6比`-23.771%`。exp413は`-38.529%`だったが、
  SP45 thread loopは`303.658秒`でv6 sequential比`+38.712%`となったため、SP45 threading単体は高速化FAILと評価する。

## 2026-08-05 v6 exact-source rerun設計

- 目的は新しい科学variantではなく、Public LB `6.541`を得たexp512について、速度最適化前のv6 packageを
  canonical kernelのlatest versionへ戻し、v6再現性gateの2回目を得ることである。
- Kaggle historical `/6` pullは403だったため、v7 push直前の存在確認で保存した
  `/tmp/exp512-v7-prepush.HuDLU8/exp512-hjyact-v2-equal-blend-inference.ipynb`を正とする。このNotebookは
  v6の52 cells / 42 code cells、embedded candidate SHA `66ed4f78...c18f804c`、model-package 5 files、
  numerical-tolerance v6 config、7 dataset metadataを保持する。
- pulled Notebook自体のSHAは`8823e6ca14d4d5c1df0a5e7d3559bd60fe4d262bf7080932280a8a44c58a09b1`。
  Kaggle normalize前のhistorical package SHA `1ff97aba...0a4cf13`とはファイル表現が異なるため、同一性は
  raw ipynb SHAではなくcode-cell SHA、embedded support manifest/candidate SHA、metadata、実行後component/
  submission SHAで判定する。
- 同じkernel id/title、GPU、internet offでpushする。slug変更、v7 sourceの手修正、別experiment作成は行わない。
- 完了後はv6実績のhjyact `b192d3f3...b9ded4a`、exp413 `3a9bbd1f...be68d87`、submission
  `b960c2b1...a713e23`との一致をprimary gateにする。不一致なら自動再pushやparameter救済を行わない。

## 共通候補生成の再利用契約

静的監査では、hjyact learned-replay blockとexp413が参照するexp073 replay sourceの間で、
`_pf_lik_allseeds`、`_beam_jit`、7-beam `beam_search`、formation/dense imputer、multi-scale NCC、
feature builder、`BEAMS`、`FORMATIONS`が同一定義・同一設定であることを確認した。一方、PF wrapperと
seed policyは一致しないため、共有境界を次で固定する。

### 共有するnode（wellごとに1回）

- raw horizontal/typewell load、ID/row alignment、既知prefix/evaluation suffix分離。
- learned-replay 7-beam bankと`beam_mean/std/median`。
- multi-scale NCC / scan候補。
- formation-plane / dense-ANCC候補と距離・分散。
- 上記から作るdeterministic GR rolling、geometry、slope、offset feature block。

各nodeは`definition SHA + input content SHA + parameters + seed policy(none) + dtype + row order`をfingerprintとし、
同一fingerprintだけをin-memory immutable cacheから再利用する。consumer adapterは列名・順序・dtypeを
source契約へ整列するだけで、候補を再生成しない。

### 共有しないnode

- hjyact SP45の`run_pf_lik_ensemble_scales`（scale 3/5/8/12、128 seeds）と14-beam selector。
- hjyact learned側likelihood-PF。GR sigma multiplier `1.3`、`seed_base=0`を保持する。
- exp413 likelihood-PF。GR sigma multiplier `1.0`、`SHA256(likpf::test::<well>)` seedを保持する。
- hjyact/exp413の`pf_ancc`・`pf_z`のうちseed policyが異なるもの。
- exp413専用exact/self-GR HMM、K16、selector/downstream model推論。
- hjyact専用guarded overlap、visible-prefix、PF seed-branch hedge。model-package nodeは無効で生成しない。

共有可能性を候補名だけで判断しない。fingerprintが異なるnodeは必ず別生成し、runtime短縮のために
パラメータやseedを寄せない。

`candidate_reuse_manifest.json`にはnode fingerprint、consumer一覧、generation count、hit count、
wall time、content SHAを保存する。production hidden runでは共有nodeを重複生成するreference pathを
実行しない。

## 実装方針（実装済み）

- 正規Notebookは直接上書きせず、まず
  `exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py`をJupytext percent形式で
  作る。採用判断までは別名候補を正とする。
- sourceを無差別に貼らず、version 2 finalへ到達する37 active code cellsと依存関数だけを抽出した。
  layer順、source定数、CSV serialization boundaryは変更しない。
- exp413はexp510 version 4で検証済みのhidden-safe dynamic regenerationを構成参照元にする。
  static visible prediction sidecarはruntime inputに含めない。
- orchestrationは`shared deterministic DAG -> route-specific stochastic nodes -> hjyact/exp413 consumers -> final`
  の一方向とし、共有nodeをconsumer内部から再計算できないAPIにする。
- source version 2でpinされた7 dataset version IDと保存model SHAをfail-closeで確認する。
- visible source-parity modeとdynamic hidden modeを同一コードに持たせ、parity modeだけが既知SHAを
  assertionする。hidden modeはrow数、well数、well IDを仮定しない。
- 特定ID、14,151行、visible 3 well、Public LB scoreを実装source内の予測分岐へ入れないことを
  AST/text regression testで確認する。

候補sourceのline数/SHAは4並列化後に再記録する。読込対象の保存modelはexp413 75ファイル、
hjyact 8ファイルの計83ファイルで、trainer wrapper内部を含む推定器は103。新規boosterは0、runtime Ridgeは5 fits。

## 検証段階

### Stage 0: 静的契約

- source/version/input SHA、active cell order、保存model inventory、seed policyをmanifestへ固定する。
- 共有candidate DAGのnode fingerprintとconsumer mapを固定し、spy/counter testで共有nodeがwellごとに
  1回だけ呼ばれることを確認する。
- dedicated testsでID alignment、nonfinite/duplicate/fallback、layer順、formula、final-write exclusivity、
  forbidden fixed-ID/static-output-copyに加え、PFの誤共有とcache key衝突を監査する。
- Jupytext `--test`、`py_compile`、Ruff F821、`validate-exp`を通す。

### Stage 1: visible source parity

- Kaggle GPU/internet offのproduction shared-DAG modeでhjyactとexp413を同時生成する。
- hjyact source final SHAはbyte-identicalをprimary gateにする。exp413 predictionはexact reference content SHA
  `875a1334...dc4`、またはv3--v5で再現しreference差max `0.0165 ft` / RMSE `0.000753012 ft`を
  監査済みのwitness SHA `3a9bbd1f...be68d87`だけを受け入れる。
- exp413 witnessの許容上限はユーザー承認どおりmax absolute `0.02 ft`、RMSE `0.001 ft`とする。
  未知の別SHA、上限超過、ID/order不一致はfail-closeする。static reference predictionはruntime入力に使わない。
- guarded overlap / visible-prefix / branch hedge reportの件数と数値署名、およびmodel-package非実行を照合する。
- reuse manifestで共有nodeのgeneration countが各well 1、2 consumerからcache hitしたことを照合する。
- exactまたは監査済みnumerical witness契約が不成立ならexp413 blendへ進まず、profile、weight、thresholdを救済しない。

### Stage 2: blend technical runと再現性rerun

- 同じcandidateでdynamic exp413とhjyactを生成し、固定blend、manifest、component readoutを保存する。
- 同一条件の2回のKaggle visible runでcomponent/final/submission SHA一致を確認する。
- 独立実行の単純上限はsource 787.8秒 + exp413約290.3秒で約18分。shared-DAG実測をnode別に記録し、
  production runtimeはこの単純合計より短くする。短縮のためにPF設定や科学条件は変更しない。

### Stage 3: 提出判断

- Stage 0--2とsubmit-checkがすべてPASSした場合だけ、別承認を得て固定1候補をcode submissionする。
- 提出後もweight/profile/閾値を変更しない。LB改善がなければnegative resultとして閉じる。
- honest OOFは存在しないため、Public LBが改善してもprivate-generalization anchorとは呼ばない。

## 再現性設計

- seed policy: source固定seedを保持。SP45 likelihood PFはseed `0..127`、visible-prefixは固定24/48 seeds、
  各active stochastic pathのseedをmanifestへ列挙する。
- stochastic 処理の有無: PF、visible-prefix PF、PF seed-branch統計にあり。
- PF/Beam / likelihood-PF / seed baggingの有無: あり。source exact replayのため数と順序を固定する。
- 並列処理と乱数の関係: SP45/HMM/PF/K16をwell単位4並列にする。active pathごとにseedを明示し、
  task completion順ではなくwell入力順で結合する。global RNGとthread scheduling依存が残らないことを
  regression testとcomponent SHAで確認する。
- CPU/GPU runtime: original version 2と同じGPUをparity基準にし、image digestを記録する。
- train cache / test regeneration: dynamic sampleから再生成し、row/well/schema/content SHAを記録する。
- shared cache: process-local・read-onlyとし、cache keyへdefinition/input/parameter/seed/dtype/orderを含める。
  consumerごとのcomponent SHAが既知referenceと一致しない場合はcacheを無効化して続行せずfail-closeする。
- model manifest: source dataset version、保存model全ファイルSHA、Ridge 5-fold設定とprediction SHAを記録する。
- gzip: decompressed content SHAを主証拠にする。
- Kaggle bootstrap: metadataとembedded configのsource version、profile、weights、seed、input IDsを照合する。

## リスク

- リークリスク: 特定IDハードコードは確認されていないが、train/test同一well contactとvisible-prefixへ強く依存する。
  visible testはPublic scoring dataではないものの、honest OOFがなくPrivate一般化は未検証。
- CV/LB不一致: hjyact `6.568`はPublic LBの単発証拠で、private性能の根拠ではない。
- 再現性: source内にはNumPy/Numba乱数とthreaded pathがある。titleだけでdeterministic anchorとせず、
  exact source parityと2-run SHA一致を必須にする。
- 実装差: active layerの一部欠落や順序変更で「hjyact final」ではなくなる。layer reportと既知SHAで防ぐ。
- hidden safety: visible output copyや固定ID sidecarはhidden rerunで破綻する。両成分をdynamic生成する。
- runtime/memory: 独立実行ではvisible約18分見込み。共有DAGでdeterministic blockを1回にするが、cacheの
  peak memoryをnode別に計測し、consumer終了後に不要nodeを解放する。hidden scalingと8GB超の入力群を
  preflightし、Kaggle上限を超える場合もPF設定や科学条件を別承認なしに縮約しない。
