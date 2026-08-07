# exp515_third_place_three_family_hmm_late_submit セッションノート

## 目的

3位チームの公開説明を基に、3種類のHMMを近似実装し、fold-safe OOFとhidden-test inferenceを実行する。
固定版を`LATE SUBMIT`と明示して1回だけ提出し、公開されたスコアとの差を確認する。

## 現在の状態

- Route: `pf_beam`
- 状態: implementation/static validation/package complete; pre-push session guard waiting
- submission phase: `post_competition_late_submission`
- CV / Public LB / Private LB: 未実行 / 未提出 / 未提出
- source code parity: 不可能（3位code非公開）

## 2026-08-07 設計

- `kaggle-review-exp`に従い、実験作成前にsteeringを作成した。
- `kaggle-platform`でKaggle CLI 2.2.3、OAuth、legacy credentialを確認した。API tokenは未設定だがCLI認証は利用可能。
- discussion `733319`をKaggle CLIで再取得し、author、本文、3-family HMM設定を確認した。
- tereka / takoihiraokazuの`rogii`公開Notebookを検索したが`Not found`で、source実装は取得できなかった。
- 元コードが非公開で、絶対TVTの状態の刻み方、遷移値、self-referenceの重み、Local-DTWの詳細を推定しているため、実装区分を`proxy`へ訂正した。特にLocal-DTWは3つの伸縮率を隠れ状態として選ぶ近似であり、元チームの処理とは一致しない。
- 実行前に固定する条件は、技術検証を通った1版だけを提出し、OOF/LB確認後に調整せず、submission messageに`LATE SUBMIT`を明記することとした。
- ユーザーには「推測したパラメータと簡略化した状態空間を使い、元コードの忠実な再現ではない」と説明済み。その後の「実行してください」を、この実装を実行する明示承認として記録する。
- `docs/06_reproducibility.md`を確認し、RNGなし、sorted order、prediction/submission/reference manifest SHAを設計した。
- 親compact構成比較: exp374 trainは10章 / 1,880行。exp515 trainは8役割章 / 1,449行、inferenceはshared exact-HMM章を含む11役割章 / 1,830行で、input、reference生成、joint HMM、orchestration、metrics、生成物をNotebook上で追える。薄いhelper呼び出し構成ではない。

## 実行量契約

- scientific variants / HMM families: `3 / 3`
- full train maximum HMM well-runs: `773 x 3 = 2,319`
- reporting folds: 5（model training foldは0）
- LightGBM config / trained fold / booster / trained model: `0 / 0 / 0 / 0`
- PF / Beam / parent-control rerun: `0 / 0 / 0`
- CPU / GPU / TPU / internet: `1 / 0 / 0 / off`
- late submission attempts: 1

## 2026-08-07 実装・検証

- canonical compact self-contained train / inferenceをJupytext percent sourceから作成し、正規Notebookへ採用した。
- hidden stateは`TVT offset x formation rate x GR bias x reference family`。base/local-DTW/fine-binの3 familyを別exact forward-backwardでdecodeし、固定`0.50/0.20/0.30`で混合する。
- same-typewell sibling atlasはtrain OOFでvalidation fold全wellを除外し、inferenceではruntime trainだけから再生成する。test typewell groupはnative GR k-gram一致、次にtarget-free cosine fallbackで動的解決する。
- inferenceはruntime sample submissionをschema/ID集合/行順の正とし、公開testのwell ID・3 wells・14,151 rowsをcodeに含めない。
- exact kernelを小型dense全状態sum-productと比較し、posterior mean / log evidenceを`2e-6 / 2e-5`以内で一致確認した。
- 専用contract `10 passed`、共通Notebook込み`15 passed`。py_compile、Ruff F821/F401/F841、Jupytext round-trip、strict `validate-exp`、template validationをPASSした。`__file__`依存は0。
- local環境にはNumbaがないため実raw one-well smokeは実行不能。小型kernelはNumba fallbackの同式で検証済みで、公式full smokeはKaggle CPUを正とする。
- canonical kernel slug/title:
  - train: `kentookumura/exp515-third-place-hmm-late-submit-train` / 40 chars
  - inference: `kentookumura/exp515-third-place-hmm-late-submit-inference` / 44 chars
- packageはprivate / CPU / GPU・TPU・internet off / run-on-push / competition source 1 / kernel source exp065 1。train/inferenceは19/25 cells、637,792/660,433 bytes。
- canonical / train package / inference package config SHAは全て`46bdbae7...cb0f6`で一致した。
- `2026-08-07 13:00:11 UTC`時点でPlaywright/Selenium/browser connectorがなく、Kaggle UIのActive Sessions `CPU active/limit`と`GPU active/limit`を取得できなかった。`kaggle quota`やkernel listで代用せず、session guard規約どおりpushしていない。

## 2026-08-07 実行前の訂正と再検証

- 元コードと一致しない近似実装であることに合わせ、実装区分を`proxy`へ訂正した。HMMの計算内容と提出前に固定した設定値は変更していない。
- 専用testは`10 passed`、`py_compile`、Ruff、Jupytext検査、strict experiment validationを再度通した。
- train/inferenceのpush用Notebookを`LATE SUBMIT`入りのtitleで再生成した。どちらもprivate / CPU / GPU・TPU・internet off / run-on-pushである。
- canonical / train package / inference package config SHAは全て`5d80e0f02dc13164101a5c90a977666dfba2c738bbaf0acfcad3995fb51ce811`で一致した。
- `2026-08-07 13:08 UTC`時点でもKaggle UIを操作できるbrowser connectorはなく、CPU/GPUの`active / limit`は未確認のためpushしていない。
- その後、ユーザーから実行指示が繰り返されたため、CPU/GPUの`active / limit`を取得できないことを説明済みのうえで、既存sessionを停止せずCPU trainをpushする指示として記録した。

## 次のアクション

1. [x] self-contained Jupytext train/inferenceとcontract testsを実装する。
2. [x] staticとsynthetic exactnessを通す。local raw smokeはNumba不足のためKaggleへ移す。
3. [ ] ユーザーからKaggle UIのCPU/GPU `active / limit`または画面画像を受け、CPU空き確認後にtrainをpushする。
4. [ ] Kaggle CPU train/inferenceを実行する。
5. [ ] submit-check後、固定messageでlate submitし記録を更新する。

## 2026-08-07 Kaggle train実行

- `kaggle kernels push -p experiments/exp515_third_place_three_family_hmm_late_submit/kaggle/train`を実行し、`kentookumura/exp515-third-place-hmm-late-submit-train` version 1を開始した。
- URL: `https://www.kaggle.com/code/kentookumura/exp515-third-place-hmm-late-submit-train`
- push用Notebookのtitleは`exp515 third place HMM LATE SUBMIT train`。CPU、private、internet offで実行中。
- `2026-08-07 13:25:38 UTC`時点でOOFは`7 / 773`坑井。1坑井あたり約29〜45秒で、単純外挿では完了まで約8時間。Kaggle側の実行は継続する。
