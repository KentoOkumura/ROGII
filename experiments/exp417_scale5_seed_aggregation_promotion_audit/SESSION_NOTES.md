# exp417_scale5_seed_aggregation_promotion_audit セッションノート

## 目的

同じexp404 x1.0 PF seed bankの算術平均と固定temperature-5 likelihood平均を、
新しいPFを回さず全OOFのfold / scope / tail gateで比較する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage A version 1 technical PASS / scientific FAIL / terminal close
- artifact parent: exp404
- scientific control: exp072
- 原因根拠: exp410
- CV: candidate `10.914522073` / control `11.594897884`
- LB: なし
- ユーザー承認範囲: 正規train Notebook採用 / package / push / Stage A runまで
- inference / submission: 未承認・未実行

## 固定した比較

```text
control   = arithmetic mean of 128 x1.0 seed trajectories
candidate = normalized exp((seed_total_loglik - max_loglik) / 5) weighted mean
```

- temperature 5.0固定
- 同一particles / seeds / trajectories
- full-suffix GRを使うtarget-free batch readout
- suffix TVTは使わない
- causal online predictionとは扱わない

## Stage A予定量

- saved candidate readouts: 1
- PF well-runs / control reruns: `0 / 0`
- model configs / trained folds / boosters: `0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`
- expected rows / wells / reporting folds: `3,783,989 / 773 / 5`

## 再現性メモ

- exp404 frozen prediction raw/decompressed/logical/schema SHAとscientific contract SHAを
  すべてmandatoryにする
- Stage A RNGなし、保存prediction readoutのみ
- prediction identity freeze前にtruth/error/fold/hidden-like roleを読まない
- Stage Aだけではinference deterministic anchorと呼ばない

## コマンドログ

2026-07-27:

- `make new-steering EXP=exp417_scale5_seed_aggregation_promotion_audit`
- `make new-exp EXP=exp417_scale5_seed_aggregation_promotion_audit SOURCE=templates/experiment`
- scaffoldとdesign-only文書を作成
- PF / model / Notebook / Kaggle実行は0

2026-07-28:

- ユーザーの`exp417を実装してください`を、設計済みStage Aのtrain-side実装承認
  として記録した。
- Jupytext percent形式のcompact self-contained train候補とfail-closed
  inference候補を作成し、それぞれ候補Notebookへ変換した。
- 正規`*_train.ipynb` / `*_inference.ipynb`はplaceholderのまま上書きしていない。
- exp404 frozen predictionのraw / decompressed / logical / schema SHA、
  exp404 well audit raw SHA、scientific contract SHAをmandatoryにした。
- exp404 well auditとcontractから、x1.0算術平均とscale5が同じ500 particles ×
  128 seeds、同じseed labels、同じGR scaleのreadoutであることをfreezeする。
- freeze後だけexp226 truth / fold、exp115 hidden-like roleを読むlate-joinを実装した。
- exp072 arithmetic meanとexp209 exact-HMMを保存生成物から読み、control RMSE parityと
  fixed HMM/LikPF 50:50 non-regressionを実装した。
- direct pooled / 5 folds / raw observed / raw missing / high-missing / 1000+ /
  hidden-like 2面 / by-well p95 / worst well / fixed blendの単一AND gateを実装した。
- Stage A実行契約はsaved candidate readout 1、scientific candidate 1、
  PF / parent PF / model config / trained fold / booster / HMM / Beam / GPU
  `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`、reporting folds 5。
- 専用contract testsは`8 passed`。共通の`test_kaggle_notebooks.py` /
  `test_scaffold.py`と合わせて`19 passed`。
- `py_compile`、Ruff全規則、Ruff format check、
  train / inference両方の`jupytext --to ipynb --test`を通過した。
- `make validate-exp EXP=exp417_scale5_seed_aggregation_promotion_audit`は
  strict validationを通過した。
- `task` executableは環境にないため、Makefile同等コマンドを使用した。
- 親compact比較: exp404 trainは`2,174`行、exp417 train候補は`1,637`行。
  exp417はPF kernelを再掲せず、saved-input preflight、same-bank freeze、late join、
  direct / blend metrics、gate、生成物を10章で追える。
- implementation SHA:
  - train source:
    `721a3201c4a1f5aab5911dbc164db56439b72a04809358bfe2d961cae9ac899e`
  - compact train Notebook:
    `0afdc884edb2d2c078a0ee33dbde796fa63ade45ccac7b851c5dcd21ea74859c`
  - inference source:
    `45c59328f5184a2680b098ea4515efdbd2efec325108808068590807a20cdf57`
  - compact inference Notebook:
    `963a42d9bb8f2976a14ac575919164540956984420ce52d5e5c584e1ed63dbe5`
  - config:
    `c6b48578868bfcb40963adc404c6d735f26b5c7ba71d2f36779bd5412b8ffc10`
  - contract tests:
    `3be208cb5a90d850c444e9b9d44acc18eb8166d94eeba773b638ee545c3d6d19`
- PF / HMM / model / Kaggle package / push / run / inference / submissionは0。

2026-07-28（Stage A実行承認）:

- ユーザーの`実行してください`を、直前に提示した正規train Notebook採用、
  Kaggle package / push、Stage A saved-artifact audit実行の承認として記録した。
- push前実行量を再確認した。
  - scientific candidates / saved candidate readouts: `1 / 1`
  - PF well-runs / parent PF control reruns: `0 / 0`
  - model configs / trained folds / boosters: `0 / 0 / 0`
  - HMM / Beam / GPU runs: `0 / 0 / 0`
  - reporting folds: `5`
- 既存controlや親実験の再学習・再生成はなく、保存済みartifactの決定論的readout
  だけをCPUで実行する。
- Kaggle CLI OAuth疎通、exp404 dataset source、exp072 / exp209 / exp226 /
  exp115 kernel source、および必要入力filenameの存在を確認した。
- 完全なslug
  `exp417-scale5-seed-aggregation-promotion-audit-train`は52文字でKaggle上限を
  超えるため、意味を維持して`aggregation`を`agg`に短縮し、
  `exp417-scale5-seed-agg-promotion-audit-train`を採用した。
- inference / submissionは今回の承認範囲外のままとする。
- 正規train Notebookをcompact self-contained sourceから採用し、strict private CPU
  packageを作成した。packageの入力にはexp404 dataset 1件と保存生成物を読む
  kernel source 4件だけを含める。

2026-07-28（Stage A完了）:

- `kentookumura/exp417-scale5-seed-agg-promotion-audit-train` version 1
  （id_no `128917131`）をprivate CPU / internet offで完了した。
- runtimeは`154.215039 sec`、rows / wells / foldsは
  `3,783,989 / 773 / [0,1,2,3,4]`。
- technical gateは全PASS:
  - mandatory input SHA一致
  - exp404 parent prediction logical SHA:
    `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`
  - exp417 Stage A prediction logical SHA:
    `62b22b65cd3f946c5d48fa0da10859180e1b5ade7f3da063f3a7d429ab5dead9`
  - same x1.0 bank: 500 particles ×128 seeds、same seed labels、
    same GR scale、temperature 5.0
  - freeze前のtruth / fold / hidden-like role / error row読取:
    `0 / 0 / 0 / 0`
  - exp072 arithmetic、exp209 HMM、fixed 50:50 control parityは
    すべて`1e-5 ft`以内
  - Stage A PF / parent PF / HMM / Beam / model config / trained fold /
    booster / GPU: 全て0
- scientific readout:
  - arithmetic control RMSE: `11.59489788373621`
  - fixed scale-5 RMSE: `10.914522073423171`
  - pooled gain: `0.6803758103130395 ft`
  - improved folds: `5 / 5`
  - raw-GR observed gain: `0.744954600209498 ft`
  - raw-GR missing / high-missing / 1000+ / hidden-like spatial /
    hidden-like typewell-purged: 全て非悪化
  - fixed exp209 HMM/LikPF 50:50 gain: `0.18478646708237534 ft`
  - direct by-well improved / regressed: `516 / 257`
- scientific gate FAIL:
  - by-well delta RMSE p95 `+2.9416884826255085 ft`
    （上限`0.0 ft`）
  - worst well `70925e23` regression `+25.311274575082955 ft`
    （上限`0.25 ft`）
- decision:
  `fixed_scale5_seed_aggregation_rejected_close_without_rescue`
- terminal close反映後は`execution.run_stage_a=false`、
  `runtime.kaggle.run_on_push=false`として、同じStage Aの偶発再実行も停止した。
- 出力archiveはSHA / manifestの実ファイル検証に必要なため一時領域だけへ取得した。
  artifact manifest 7件のbytes / SHAは全一致した。
  - scientific contract SHA:
    `715e755e625281980cb9573f5f905d4a326ab64844d64fbe8b3be7a27978b7e9`
  - input manifest SHA:
    `704d417144f5f7da55748f4c517a70b8e80d2755712fa1f3a2adbbfa2c79e445`
  - artifact manifest SHA:
    `375a65e3463a4313707a76b670616c58e22bd74774c5b1a18f3e1a180a7e7078`
- 事前契約どおりtemperature / scale / best seed / median / mode / medoid /
  selector / well-row gateのsame-OOF救済は行わず、inference / submissionも
  実行しない。exp413の独立ML branch判断も変更しない。
- executed package / final local record SHA:
  - Kaggle version 1 package Notebook:
    `9d6c04a4a5d38dfe6b42b288c76aea9e42db0b749115f98f7ac5ea5cda49ff98`
  - canonical train Notebook:
    `2a588fa3689670ee960fdc7f90272a6a895f05564b2baa31b23a2032800520de`
  - final config:
    `e8aa576b031b08e852b83599e9b51470b574f76f0078428e7aed267dd71ba33b`
  - final metrics:
    `84c220651248d495471ab82347bf116bece3130deba55f54f89e946b673fe4f8`
  - final contract tests:
    `b417a85723de1fb51c8ef417241c360c244ff9bbfb43b391c50f537ba07c971c`
- 最終検証は専用+共通`19 passed`、py_compile、Ruff、Jupytext roundtrip、
  strict experiment validationを全てPASSした。

## 次のアクション

exp417はStage A scientific FAILでterminal close。追加実行、raw-test inference、
submission、same-OOF救済は行わない。
