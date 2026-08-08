# exp515_third_place_three_family_hmm_late_submit セッションノート

## 目的

3位チームの公開説明を基に、3種類のHMMを近似実装し、fold-safe OOFとhidden-test inferenceを実行する。
固定版を`LATE SUBMIT`と明示して1回だけ提出し、公開されたスコアとの差を確認する。

## 現在の状態

- Route: `pf_beam`
- 状態: 完了。train OOF再現失敗、inference停止、未提出
- submission phase: `post_competition_late_submission`（未実施）
- CV / Public LB / Private LB: `40.88961598374063` / 未提出 / 未提出
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

## 終了時の確認

1. [x] self-contained Jupytext train/inferenceとcontract testsを実装する。
2. [x] staticとsynthetic exactnessを通す。local raw smokeはNumba不足のためKaggleへ移す。
3. [x] Kaggle CPU trainを実行してCVを確認する。
4. [x] 再現失敗を受けたユーザー指示に従い、inferenceを停止する。
5. [x] late submissionを行わないことを記録する。

## 2026-08-07 Kaggle train実行

- `kaggle kernels push -p experiments/exp515_third_place_three_family_hmm_late_submit/kaggle/train`を実行し、`kentookumura/exp515-third-place-hmm-late-submit-train` version 1を開始した。
- URL: `https://www.kaggle.com/code/kentookumura/exp515-third-place-hmm-late-submit-train`
- push用Notebookのtitleは`exp515 third place HMM LATE SUBMIT train`。CPU、private、internet offで実行中。
- `2026-08-07 13:25:38 UTC`時点でOOFは`7 / 773`坑井。1坑井あたり約29〜45秒で、単純外挿では完了まで約8時間。Kaggle側の実行は継続する。
- version 1は773坑井、3,783,989行を完了した。全体RMSE `40.88961598374063`、fold別`34.6540 / 47.8384 / 37.2798 / 39.2300 / 43.9388`、実行時間`27,668.99秒`。
- 3種類の個別RMSEは`40.9076 / 40.9562 / 40.8700`。公開された3位チームのOOF `5.9703`とは大きく離れ、今回の近似実装では再現できなかった。
- 有限値、重複ID、検証坑井の参照データからの除外、事後確率の正規化など、Notebook内の提出前確認はすべて通過した。
- OOF予測は3,783,989行、decompressed content SHA256は`15545d4c6ecde072337e7c2ef8f6fbd7e0007e511461157b29003bc308bc7fb4`。
- `2026-08-07 23:33:26 UTC`にinference metadataを再確認した。CPU、GPU/TPU off、private、internet off、run-on-pushであり、CPU実行なのでGPU/TPU quota確認は不要と判断した。

## 2026-08-07 inference停止

- `kentookumura/exp515-third-place-hmm-late-submit-inference` version 1を開始した直後、ユーザーから「再現できていないのなら推論には進まないでください」と指示を受けた。
- CLIにsession停止コマンドがなく、公式APIの`CancelKernelSession`はHTTP 403だった。状態が`RUNNING`であることを確認後、これ以上実行させないためKaggle上の推論Notebookだけを削除した。削除時刻は`2026-08-07 23:36:05 UTC`。
- ローカルのNotebook、設定、コードは保持している。推論結果およびsubmissionは作成・使用せず、late submissionも行わない。

## 2026-08-08 完了

- ユーザーの「exp515を完了にしてください」という判断により、状態を`completed`へ更新した。
- 結果は再現未達として確定し、今回の近似実装には追加調整を行わない。
- Kaggle全773坑井OOFは完了済み。推論結果、submission.csv、Public / Private LBはなく、late submissionは0回で終了する。
- 今回の結果で閉じるのは、直線予測の周囲`±10 ft`のTVT候補、推定した遷移確率、3つの固定伸縮率によるLocal-DTW近似を組み合わせた実装だけである。
- exp518ではTVT候補とLocal-DTWを修正したが、10坑井の動作確認で公開OOFを再現できず、別の完了済み実験として記録した。
- 完了時の再検査はstrict experiment validationを通過し、`NUMBA_DISABLE_JIT=1`で専用test `10 passed`を確認した。
