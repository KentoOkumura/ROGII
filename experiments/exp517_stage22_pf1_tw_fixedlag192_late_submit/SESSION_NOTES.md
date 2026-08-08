# exp517_stage22_pf1_tw_fixedlag192_late_submit セッションノート

## 2026-08-08 v2修正（現行）

- ユーザー指摘どおり、v1の`1 PF direct mean`はstage 2-2再現ではない。旧「技術的再現PASS」を撤回し、契約不一致FAILとして同一exp内に保持する。名称変更や別実験への付け替えは行わない。
- 修正版v2は`pf_1 / pf_2 / pf_3 / r0_seed32 / r1_seed32 × twGR × 32 seeds × fixed-lag 192`をすべて生成し、公開Ravaghi tabular stackへ連結する。
- stage 2-2後のStudent-t likelihood、tempering、ps-combo、GR-free anchor、learned emission、self/nbr、whole smoothingは明示的に無効化する。
- targetは`TVT-last_known_tvt`。3 LightGBM + 2 CatBoost × 5 GroupKFold = 25 base models、positive Ridge 5 folds、control再学習0。
- decodeは公開Notebookどおり`0.91 model + 0.09 pf_1`、tau 85 fade、testでSG(17,3)。
- `scripts/prepare_exp517_stage22_fivepf_tabular.py`からv2 compact self-contained train/inference候補を生成した。v1 source SHA `f01b0114...b29`はcontract testで固定し、未変更を確認する。
- generator `--check`、py_compile、Ruff F821、v2 contract test 6件、Jupytext変換/pairingはPASS。
- v2実装検証時点ではKaggle train未実行。published CV `7.50`との同条件gateを通るまでinference/LATE SUBMITへ進まない。

### Kaggle train v1 push

- 2026-08-08 00:00:14 UTC、`kentookumura/exp517-stage22-5pf-fl192-tab-train` version 1をpushし、`RUNNING`を確認した。
- push直前に表示されたquotaは旧週のGPU残量`0.70h`だったが、ユーザーの即時push指示に従って実行した。push直後に週次更新され、GPU quotaはused `0.00h` / remaining `45.00h` / refresh `2026-08-15T00:00:00`となった。
- 実行対象はscientific variant 1、5 PF banks、3 LightGBM configs、2 CatBoost configs、5 folds、25 base models、5 positive Ridge models、control/parent rerun 0。
- CV gate未判定。inferenceおよびLATE SUBMITは未実行。

### Kaggle train v1失敗 / v2修正push

- train version 1は約7秒で`ValueError: No kernel name found in notebook and no override provided.`となり、学習セルへ入る前に失敗した。科学的結果は生成していない。
- 原因はJupytext生成Notebookのtop-level metadataに`kernelspec`がなかったこと。self-contained source生成元へ`Python 3 / python3` kernelspecを追加した。
- 再生成後、v2 contract test 6件、generator check、Ruff F821、Notebook内kernelspec存在、`git diff --check`をPASSした。
- 同じslugへversion 2をpushし、`RUNNING`を確認した。手法契約、実行variant数、PF/model数はversion 1から変更していない。

### Kaggle train v2完了 / CV再現判定

- ユーザーGUIの完了連絡後、`kaggle kernels logs -f kentookumura/exp517-stage22-5pf-fl192-tab-train`とCLI statusでversion 2 `COMPLETE`を確認した。
- runtimeは`14,603.468 s`。Tesla T4 x2、3,783,989 rows、773 wells、280 features。
- 5 PF bankは全773 wellsを生成。bank別秒は`pf_1 922.418 / pf_2 1002.812 / pf_3 905.882 / r0 588.493 / r1 636.428`、PF合計`4,145.890 s`。
- 25 base models（LightGBM 15 + CatBoost 10）とpositive Ridge 5を保存し、model manifestにfeature順序、config/fold、relative path、SHAを記録した。control rerunは0。
- Ridge OOF RMSE `7.531449305`、public postprocess OOF RMSE `7.536732106`、SG diagnostic `7.536165229`。
- published stage 2-2 CV `7.50`との差は`+0.036732106`（0.49%）。score regime再現はPASS、exact numeric parityは未達として区別する。
- OOF decompressed-content SHA256は`31ce9041decf340de0f91a0d33de86bb68b35541e9d995158e17dcd6a3d695bc`。
- `kaggle kernels files --page-size 200`でmetrics、manifest、15 LightGBM、10 CatBoost、5 Ridge、OOFの存在を確認した。小さいmetrics/manifestだけを`/tmp/exp517-stage22-v2-result`へ選択取得し、output archive全体は取得していない。
- CV gate通過。次は同じexp517のinference v2を固定model manifestから実行し、submit-check後に承認済みLATE SUBMITへ進む。

### inference v2 push前guard

- 2026-08-08 04:13:07 UTC、対象resourceは`NvidiaTeslaT4` GPU。quotaはused `4.06h` / remaining `40.94h` / total `45.00h` / refresh `2026-08-15T00:00:00`で、公開3井のinferenceに十分と判断した。
- scientific variant 1、5 PF banks、保存済み25 base + 5 Ridgeを読み込む。学習0、control/parent rerun 0。
- contract test 6件、py_compile、Ruff F821、Jupytext pairing、canonical/package Notebook byte一致、kernel metadataのGPU/internet/kernel sourceをPASSした。
- Active Sessions数はKaggle CLI非対応のためpush gateにしない。GPU同時上限2、CPU上限5の運用は維持する。

### inference v1 push

- `kentookumura/exp517-stage22-5pf-fl192-tab-infer` version 1を`--accelerator NvidiaTeslaT4`付きでpushした。
- push後に同じcanonical slugを`kaggle kernels pull -m`し、`enable_gpu=true`、`enable_internet=false`、`machine_shape=NvidiaTeslaT4`のKaggle反映を確認した。
- 現在inference実行中。LATE SUBMITは未実行。

### inference v1完了 / submit-check

- inference version 1はexit 0で完了。公開sampleは3 wells / 14,151 rows / 280 features、5 PF合計`55.618 s`。
- 保存済みtrain v2 model manifest SHA256 `6e8dc29cda95628d98df17ca9e84d0c698d7c0b4dbe69e205a4abd0365d881f0`を検証し、25 base + 5 Ridgeを読み込んだ。
- runtime manifestはduplicate ID 0、missing 0、finite、sample order exact。submission SHA256 `7a89df26198d04a7419c166a2f645f6b8a376d9d0924e4ffc3b7bffaa097ae45`、candidate content SHA256 `661bd05214e9ae9bfac3ecfdd713fb0f97806556c83cbb1c40c5e765273d810b`。
- Kaggle outputから`submission.csv`とexecution manifestだけを選択取得した。`kaggle-submit-check`はCSVとNotebook metadataの両方でFAIL 0 / WARN 0。ローカルsampleとのcolumns、ID値・順序、14,151行も完全一致した。
- hidden testはruntime competition root/test/sampleを動的解決し、well/row/ID/SHAを固定しない。保存済みmodel manifestとhidden inputだけで280特徴量を再生成し、runtime sampleへIDでone-to-one整列する。
- 提出前判定: PASS。承認済みの固定LATE SUBMITへ進む。

### v2 LATE SUBMIT受付

- 最初のCLI callはcode competitionで必要な`-f submission.csv`を付けず、`CreateCodeSubmission 400`となった。Kaggle側のsubmissionは作成されていない。
- 同じcanonical kernel/versionへ`-f submission.csv`を追加して再実行し、ref `55340618`を受付。別slug、別実験、別予測への変更はない。
- submitted at `2026-08-08 04:18:59.837000 UTC`、status `PENDING`。pollingログは一時生成物としてGit非追跡で保持する。
- 22分時点でも`PENDING`。ユーザー指示によりローカルmonitor processだけを停止した。Kaggle側のscoring submission ref `55340618`はcancelしていない。

## 目的と現在の状態

- 目的: stage 2-2の`5 PF × twGR × fixed-lag 192`を公開Ravaghi tabular stackへ入力し、公開値CV `7.50`の再現を確認してから`LATE SUBMIT`する。
- Route / fidelity: `ensemble / historical_contract_reconstruction`
- v1状態: `contract_mismatch_failed`。PF単体 Public `7.825` / Private `9.689`は失敗履歴。
- v2状態: 学習、推論、submit-check、LATE SUBMIT、公式scoring終了。CVとPublicは近似再現、Privateは未再現。ユーザー判断によりstatusは`completed`。
- Source: discussion 733226 stage 2-1/2-2、public kernel id_no `126919690`、scriptVersionId `340359218`
- 公開Notebook SHA: `b44f7889d6abdf9b027d33cb6c6b45f23902d609fae8d06f332914017784c924`
- 公開v96 config raw SHA: `80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3`

## 手法契約とproxy承認

- stage 2-2掲載値 CV `7.50` / Public `6.724` / Private `7.404`は`5-input + smooth`をtabular modelへ入れたsystemの値で、PF単体値ではない。
- writeupはGPU bootstrap PF、32 seeds、fixed-lag particle smoother、lag 192、および`pf_1 × twGR`のfilter/smoother例を公開している。
- v1は最終公開v96 `pf_1`だけを使い、tabularを省略して直接decodeした`proxy`だったため、stage 2-2再現として無効である。
- v2は公開された5 bank configとRavaghi tabular source/splitを接続して手法契約を修復した。Kaggle CVで掲載値近傍を確認するまでは再現完了としない。

## 実行量契約

- scientific variants: 1
- PF banks / representations: `5 / 1`
- particle trajectories per well: `(600 + 600 + 600 + 400 + 400) × 32 = 83,200`
- smoothing: `fixedlag / 192`
- LightGBM configs / CatBoost configs / folds: `3 / 2 / 5`
- base models / positive Ridge models / parent-control rerun: `25 / 5 / 0`
- late submission attempts: 1

## 再現性

- PF seed: `4423098`
- runtime: Kaggle T4 x2、float32、internet off
- public config/source、runtime override、well/row、candidate content、prediction、submission SHAをmanifestへ保存する。
- 公開GPU実装はchunk/device分配とGPU reductionで揺れうるため、rerun一致前にdeterministic anchorと呼ばない。
- raw hidden testをruntimeから動的列挙し、runtime sample submissionをschema/order/ID集合の正とする。

## 2026-08-07 設計・実装

- `kaggle-review-exp`の手法忠実性ガードに従い、steeringへ`input / target / output / loss / decode / context unit`を固定した。
- exp516を構造parentとしてexp517を作成した。exp516の保存済みlate result `10.056 / 8.552`は再学習・再実行せず参考値に限定する。
- 公開NotebookからPF engineだけをSHA固定抽出するself-contained generatorを作成した。
- runtime overrideは`pf_1 / tw / 600 / 32 / fixedlag / lag192 / physics=false / emission=0`の1本。空anchor payloadは公開moduleのimport要件だけを満たし、weightには使わない。
- 親compact版は9章・1,265行、新compact版は不要なanchor/emission章を除いた5章・981行。Imports、runtime/input guard、PF engine、single-route orchestration、metrics/manifest/outputをNotebook上で追える。
- compact source SHAは`f01b011475a2c205658e06edac9df6ba435e9296590e4561b75ff40053113b29`、canonical/compact Notebook SHAは`4a617d221734e5b9d19e0416aa038104499707092062c091fb3f184204d1fd27`。
- generator `--check`、py_compile、Ruff F821、contract test `7 passed`、Jupytext pairing、strict experiment validation、template validationをPASSした。
- canonical Kaggle slug/titleを`exp517-stage22-pf1-tw-fl192-late-infer` / `exp517 stage22 pf1 tw fl192 late infer`へ固定した。slug長38、private、T4、GPU有効、TPU/internet無効、run-on-push有効、competition/public-kernel source付き。
- package config/sourceは正のファイルとSHA一致。config SHA `bf34cac3...ff18`、compact source SHA `f01b0114...b29`、push Notebook SHA `4c02d998...b518`、metadata SHA `e89a7651...33f0`。
- push直前`2026-08-07 15:04 UTC`のGPU quotaはused `44.29h` / remaining `0.71h` / total `45.00h` / refresh `2026-08-08 00:00 UTC`。対象はT4 GPU。anchor/emission/MLが0で、writeupの全773井fixed-lag192はRTX 5070で約13分のため、T4差を含めても42.6分内で完走可能と判断して承認済み固定runを進める。
- Active Sessions数はCLI非対応のためpush gateにしない。GPU同時上限2、CPU上限5は維持し、上限error時だけ既存sessionを無断停止せず対応を確認する。

## 2026-08-08 Kaggle public commit実行・提出前検査・LATE SUBMIT

- `kentookumura/exp517-stage22-pf1-tw-fl192-late-infer` version 1をT4 x2、internet offで実行し、exit 0で完了した。
- 公開sample実行は3 wells / 14,151 rows。実行約は`pf_1 / tw / 600 / 32 / fixedlag / lag192 / physics=false / emission=0`の1本で、PF runtimeは`12.177876968 s`だった。このwell数・runtime・出力SHAはpublic commit runの実測であり、hidden rerunの実測ではない。
- runtime contractは`id,tvt` / 14,151 rows / duplicate 0 / missing 0 / finite / sample ID order exact。`kaggle-submit-check`もFAIL 0 / WARN 0でPASSした。
- `submission.csv` SHA256は`ca9777cf782603f8cedfa4812b5762922015d8f43b10df345cee0cdb4ae2bb8d`。candidateのdecompressed-content SHA256は`88aa41f1d9c510f649e6a9bd22ed9260ec26ff9866f16ce301b908b17afaa23d`。
- 同じkernel version 1を`LATE SUBMIT | exp517 | stage2-2 pf1 x twGR fixedlag192 proxy | fixed v1`で1回だけ提出した。submission refは`55327703`。LB後の再提出は行わない。

## 2026-08-08 hidden rerun確定

- Kaggle CLI monitorとユーザーのGUI確認でref `55327703`の`COMPLETE`を確認した。Public `7.825` / Private `9.689`、scoring 10分。
- stage 2-2掲載の5 PF + tabular system `6.724 / 7.404`との見かけの差はPublic `+1.101 ft`、Private `+2.285 ft`。契約が異なるため、この差は性能再現失敗の同条件比較ではない。
- exp516 final-v96 `pfA × twGR` whole-smoother proxy `10.056 / 8.552`比ではPublic `-2.231 ft`改善、Private `+1.137 ft`悪化。stage 2-2型fixed-lag系のprivate優位はこの単体proxyでは確認できない。
- 技術的にはhidden-compatible PF生成とlate scoringを完了した。閉じる範囲は`(final-public pf_1 parameter, twGR, GR-only observation, standalone fixed-lag-192 mean decode, no fusion, hidden late rerun, 600×32, T4x2)`のみ。stage 2-2 exact 5-PF + tabular system、PF family、final 6th-place systemは閉じない。

## 次のアクション

- 固定したv2のone-shot LATE SUBMITまで完了したため、LB後調整や再提出は行わない。
- Private掲載値との差`+0.412 ft`は未解決事項として残し、stage 2-2全体の完全再現は主張しない。

## 2026-08-08 v2 hidden rerun確定

- ユーザーのGUI完了連絡後、Kaggle CLIの一回確認でref `55340618`が`COMPLETE`、Public `6.778`、Private `7.816`であることを確認した。
- stage 2-2掲載値はCV `7.50`、Public `6.724`、Private `7.404`。v2との差はCV `+0.036732`（`+0.49%`）、Public `+0.054 ft`（`+0.80%`）、Private `+0.412 ft`（`+5.56%`）。
- 契約不一致v1 ref `55327703`からはPublic `-1.047 ft`、Private `-1.873 ft`改善した。
- CVとPublicは近似再現、Privateは未再現。したがってPrivateを含むstage 2-2全体のLB再現は未達と判定し、名称変更や別結果への付け替えは行わない。
- monitorはユーザー指示により22分時点で停止していた。`COMPLETE`確認時刻は`2026-08-08 05:40:04 UTC`だが、正確なscoring所要時間は不明。
- one-shot提出を消費済み。LB後調整、再学習、再提出は行わない。
- 「scoringが完了した」というユーザー連絡を実験の完了判断と解釈してはならない。statusは`running`のまま保持し、完了・採否は別途ユーザー判断を待つ。
- その後、ユーザーがexp516とexp517を明示的に完了と判断したため、未再現判定を維持したままstatusを`completed`へ変更した。
