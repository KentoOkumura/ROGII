# exp516_sixth_place_pfa_tw_late_submit セッションノート

## 目的

6位解法の公開sourceから単体`pfA × twGR`を忠実再現し、固定版を`LATE SUBMIT`と明記して1回だけ提出する。

## 現在の状態

- Route: `pf_beam`
- 状態: faithful component実装・static/package検証完了、Kaggle GPU quota refresh待ち
- Source: discussion 733226 / public kernel id_no `126919690`
- Submission phase: `post_competition_late_submission`
- CV / late LB: なし / 未提出
- 公式run: 1 bank × 1 representation × 600 particles × 32 seeds
- anchor: 5 folds × 3 seeds = 15 GRU models
- learned emission: 公開checkpoint 5本、再学習0
- LightGBM config / fold / booster / parent control rerun: `0 / 0 / 0 / 0`

## 2026-08-07 設計・source取得

- `kaggle-review-exp`、`kaggle-notebook-fetch`、`kaggle-platform`に従って一次資料と公開Notebookを確認した。
- discussion記載の旧slugは404だったため、Kaggle検索から現行slug`k256net/public20th-private6th-pf-pf-pf-pf-and-bagging`を特定した。
- 公開Notebookを`docs/notebooks/.../solution_6th/`へ保存した。source SHAは`b44f7889...c924`。
- kernel outputからv96/v97/v100のPF config JSONだけを取得した。今回使うv96 SHAは`80e973d5...02f3`。
- learned-emission checkpoint 5本を公開kernel outputから一時取得し、各4.7 MB、SHAを`config.yaml`へ固定した。
- GR-free anchorの公開training lossはsource確認によりmasked Huber（delta 8 ft）へ契約を訂正した。
- ユーザー確認により、91候補+NN融合ではなく単体PFへscopeを固定した。
- 手法契約は`pfA × twGR`、GR-free anchor、learned emission、600 particles、32 seeds、whole-interval smoother、seed likelihood decode。実装区分は`faithful`。
- late-submit契約は固定版1回、LB後retuneなし、message/title/docへ`LATE SUBMIT`明記とした。

## 実行量契約

- scientific variants: 1
- PF banks / representations: `1 / 1`
- per-well particle trajectories: `600 × 32 = 19,200`
- anchor GRU trained models: 15
- public encoder loaded models: 5
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- parent/control reruns: 0
- late submission attempts: 1

## 再現性メモ

- PF generation seed: `4423098`
- stochastic components: anchor GRU、PF transition/jump/resampling
- target runtime: Kaggle T4 x2 / internet off / float32 / full smoother
- source/config/checkpoint/prediction/submission SHAをmanifestへ保存する。
- GPU replayは同一environment rerun一致前にdeterministic anchorと呼ばない。

## 2026-08-07 実装・静的検証

- 公開Notebookの`gen_grfree_anchor.py`、`nn_emission_v97.py`、`pf_banks_v95.py`から必要セルを機械抽出するgeneratorを追加した。
- compact self-contained source SHA: `7aca22b...ccd2a6`。
- `pfA × twGR`だけを実行する固定orchestration、dynamic input/sample schema、checkpoint SHA fail-close、T4 x2 fail-close、model/execution manifestを実装した。
- contract tests: `6 passed`。構文、Ruff F821、generator `--check`、Jupytext round-trip、strict experiment validationもPASS。
- Kaggle packageを`kentookumura/exp516-sixth-pfa-late-submit-inference`、private、T4、internet off、public kernel source付きで生成した。
- canonical Notebook SHA: `db5bf126...58e26`。push package Notebook SHA: `5c688b75...62445`。metadata SHA: `ee0f8ddc...bd9b`。
- ユーザー指示によりActive Sessions確認をpush前gateから廃止した。アカウント上限はCPU 5、GPU 2だが、CLIでactive数を取得できないためpush前確認には使わない。
- `2026-08-07 13:17 UTC`の`kaggle quota --format json`はGPU used `43.93h`、remaining `1.07h`、total `45.00h`、refresh `2026-08-08 00:00 UTC`。公開writeupのPF単体計測はRTX 5070で全773井・fixed-lag 192が約13分だが、exp516は追加で15本のGRU再学習とfull smootherを含み、T4で1.07時間以内の信頼できる上限がない。このため途中quota枯渇を避けてpushを延期した。
- Active Sessions規約とquota status更新後のpush package Notebook SHAは`0a13473b...20cc6`。metadata SHAは`ee0f8ddc...bd9b`。
- その後ユーザーが`実行していいです`と明示し、GPU remaining 1.07hで途中停止するリスクを承知した今回の1回のpushを承認した。scientific variant 1、PF bank / representation `1 / 1`、600 particles、32 seeds、anchor GRU `5 folds × 3 seeds = 15 models`、public encoder 5本、LightGBM / booster / parent control rerun `0 / 0 / 0`の固定契約は変更しない。
- push直前`2026-08-07 13:20:57 UTC`のquotaはGPU used `43.93h` / remaining `1.07h` / total `45.00h` / refresh `2026-08-08 00:00 UTC`、TPU remaining `20.00h`。対象resourceはprivate Kaggle T4 GPU、internet off。ユーザーの明示上書き承認により、quota枯渇リスクを記録したうえでcanonical kernelへpushする。
- `2026-08-07`にcanonical kernel version 1をpushした。kernel `kentookumura/exp516-sixth-pfa-late-submit-inference`、id_no `129988663`。push後pullで`is_private=true`、`enable_gpu=true`、`enable_tpu=false`、`enable_internet=false`、`machine_shape=NvidiaTeslaT4`、competition/public-kernel sourceを確認した。
- 初回`kaggle kernels logs -f`は約54秒stdoutなしの後、logs stream endpoint 500で終了した。pullでkernel存在とmetadata反映済みのため、slug変更・再pushはせず同じversion 1のlive SSEを再接続する。
- 再接続したversion 1はbootstrap後、予測開始前に`RuntimeError: embedded public v96 config SHA drift`で停止した。vendor JSON raw fileはCRLF、Python`read_text()`でNotebookへ埋めた文字列はLFへ正規化されるのに、raw-file SHAをembedded-text SHAとして比較したことが原因。PF parameter、anchor、emission、decodeには到達しておらず、科学的結果はない。
- 技術修正ではraw vendor file SHA `80e973d5...02f3`とembedded normalized text SHA `aff2bcf6...d962`を分離し、runtimeは後者、lineageは前者を検証する。数値JSON値は変更なし。config SHA分離のcontract testを追加して`7 passed`、py_compile、Ruff F821、generator check、Jupytext pairing、strict validationをPASSした。
- version 2候補のsource SHA `7084348f...f460`、canonical Notebook SHA `814ae135...0cb87`、push package Notebook SHA `38b990f5...db113`、metadata SHA `ee0f8ddc...bd9b`。
- version 2 push直前`2026-08-07 13:27:01 UTC`のGPU quotaはused `43.94h` / remaining `1.06h` / total `45.00h` / refresh `2026-08-08 00:00 UTC`。version 1の消費は約0.01h。ユーザー承認済みのquotaリスクを維持して同じcanonical kernelへpushする。
- canonical kernel version 2のpushに成功した。pull後もid_no `129988663`、private、T4、internet offを維持し、Kaggle側Notebookに`PUBLIC_CONFIG_TEXT_SHA256`とtext SHA比較が含まれることを確認した。

## 次のアクション

1. `2026-08-08 00:00 UTC`以降に`kaggle quota --format json`でGPU refreshを確認する。
2. 固定inferenceをpushし、output取得・submit-check後に1回late submitする。
3. scoring監視後、結果・SHA・runtimeを公式記録へ反映する。
