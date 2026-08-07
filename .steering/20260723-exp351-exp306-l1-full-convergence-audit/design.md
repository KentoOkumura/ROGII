# 設計

## アプローチ

exp306 Stage 0で唯一full-eligibleになったL1設定を再選択せず、そのまま全773 wells / 1,546 seriesへ展開するtarget-free technical auditとする。目的はsolver収束と再現性の確認だけであり、denoised outputの科学的品質は評価しない。

exp306の固定64-well subsetはfull入力に含まれるため、新しいfull runのsubsetを抽出して親Stage 0のinput/output/status SHAと比較する。これにより追加solver rerunなしで、別Kaggle run間のexact parityを検証する。

## 実験範囲

- 対象実験: `exp351_exp306_l1_full_convergence_audit`
- Route: `pf_beam`
- 親実験: `exp306_robust_rts_l1_convergence_calibration_audit`
- 祖先: `exp304_gr_denoiser_emission_separability_readout`
- 変更する変数: 監査対象を固定64 wellsから全773 wellsへ拡張することだけ。
- 固定する変数: raw input/common preparation、missing policy、L1 objective/lambda/rho/tolerance/max iterations、series order、technical gate、thread/runtime契約。
- 除外: RTS、solver parameter search、truth join、scientific score、MRR/top3/RMSE、HMM/PF/Beam、prediction、inference、submission、exp304 selected SWT更新。
- 新規実験へ分離する理由: ユーザーが別実験ディレクトリを指定し、評価条件もStage 0 branch selectionから773-well full auditへ変わるため。solver仮説や設定は親から変更しない。

## 親anchor契約

実装時はKaggle kernel sourceとして`kentookumura/exp306-rts-l1-convergence-calibration-audit-train`を参照し、version 1由来の次のanchorをsolver実行前に検証する。kernel sourceが将来更新されても、SHA不一致ならfail-closeする。

| anchor | 固定値 |
| --- | --- |
| parent kernel version / id_no | `1` / `128231380` |
| raw well identity content SHA | `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32` |
| scientific contract file SHA | `a13bd5a7ff2119e002bfe6f8bae08207e4b2c45c9e8be0de581c27045921ee54` |
| scientific contract content SHA | `f900ae79106dbce54caf5be597afaab3d26582e08f2df5390507a534d2997915` |
| Stage 0 gate file SHA | `cdfd1397425d98076d0c4da029b5bac6640f50bff1f935ae46019024077e3887` |
| Stage 0 summary file SHA | `1217039de7d5db45c6e5d2ab9a207c555c673e662bf7bf74e82825321449e6ea` |
| 64-well sample manifest raw/content SHA | `67508cba8dab2de14e13d77edec6b8faadab8fdacd44334ca2ce029b6ddcf691` / `a483672eb47b429018e2d720289e16a05e6f6cd10c5c18213f1ba8676d04ae3c` |
| Stage 0 prepared input content SHA | `3eb28b189cd77b3d48f9745dcf49e2f8587551abfed8dcdc674101f5b1f406c8` |
| Stage 0 L1 output content SHA | `186d9682147563fe4cf1609a067004460f2e3b250a8026b7611dd712db0cbf42` |
| Stage 0 L1 status content SHA | `7b3a292ff99f8cb12abfd5917893f7d0f7b3a99109d312c0b66ae3b5966edd89` |
| 8-well parity output/status/iteration SHA | `d0b4a19788e7a13df9603b566eca35dba998b0063fb7f9e25e64b2b0b4fedec0` / `3535381302938a7a467f78d0b2fe45bb8ca587e1db9091a3a730e6262197724f` / `6488af59de4ad6ac28eb5d68b3407f5a45ea67bb42a23b051834ae3ca834b036` |

## 入力契約

1. competition raw trainからhorizontal/typewellのpaired well IDsを列挙する。
2. well IDsを昇順に正規化したraw identity manifestを作り、773 wellsと固定SHAを確認する。
3. horizontalは`MD/GR/TVT_input`だけを`usecols`で読み、`TVT/error/abs_error/formation`が処理frameに入ったら停止する。
4. typewellは`TVT/GR`だけを読み、TVT sort、GR ffill/bfill、endpoint holdを含むexp306 preparationを固定する。
5. horizontal/typewellを`well_id, series_kind, position`のstable mergesortでcanonical化する。
6. prepared inputはfinite、series長、座標order、well/series coverageを検証してからsolverへ渡す。

## 固定L1 solver契約

- branch名: `l1_iter2000_rho1_tol1e4`
- objective: `0.5 * ||y-x||_2^2 + lambda * ||D2 x||_1`
- lambda: exp306と同じfirst-difference MAD、series length、`0.67448975`、`sqrt(2 log n)`から決まる式。
- solver: second-order L1 trend ADMM。
- `rho=1.0`。
- `maximum_iterations=2000`。
- absolute / relative tolerance: `1e-4 / 1e-4`。
- adaptive rho、warm-start別解、lambda/rho/tolerance/iteration変更なし。
- exception時にfallback outputを生成しない。statusへerrorを残し、branch全体をFAILにする。

## 実行量

- active branch: 1。
- wells: 773。
- series per well: horizontal + typewellの2。
- L1 solver series-runs: `1,546`。
- Stage 0 control再実行: 0。親Stage 0 outputをSHA anchorとして参照する。
- full rerun / parity rerun: 0。full run内の64-well/8-well subsetを親SHAと比較する。
- model / LightGBM config / trained fold / HMM / PF / Beam / booster / parent control retraining / GPU: `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`。
- runtime: Kaggle CPU、internet off、worker 1、BLAS threads 1、hard limit `30,600 sec`。

## Cross-run parity

full input/output/statusをcanonical順で構築した後、親sample manifestの64 wellsを同じ順序で抽出する。

- input subsetは親Stage 0 prepared input content SHAと比較する。
- output subsetはbranch名を含む親L1 output content SHAと比較する。
- status subsetは親L1 status content SHAと比較する。
- sample先頭8 wellsはoutput/status/iteration SHAを親parity manifestと比較する。
- dataframe content SHAの列順、dtype、numeric raw bytes、string改行区切りはexp306実装と同じにする。
- 1つでも不一致なら数値再現性FAILとし、丸めや許容誤差で救済しない。

## Full technical gate

次のANDを満たす場合だけ`full_technical_pass=true`とする。

1. parent anchor SHAが全件一致。
2. raw identityが773 wells・固定SHA一致。
3. prepared inputが1,546 seriesを過不足なく持つ。
4. solver statusが1,546 rows、duplicate 0。
5. converged / technical passが`1,546/1,546`。
6. finite input/output、length/order identityが全件true。
7. silent fallback 0、error non-empty 0。
8. 64-well cross-run input/output/status SHAが完全一致。
9. 8-well output/status/iteration SHAが完全一致。
10. wall timeが`30,600 sec`以内。
11. truth/scientific score loaded false、prediction/submission null。

FAIL時は`full_technical_fail_closed`として終了する。platform/codeのpre-solver ERRORだけは同一contract修復後の再実行余地を残すが、solver gateを見たparameter変更は許可しない。

## 生成物契約

- `exp351_exp306_l1_full_convergence_audit_scientific_contract.json`
- `exp351_exp306_l1_full_convergence_audit_parent_anchor_manifest.json`
- `exp351_exp306_l1_full_convergence_audit_raw_identity_manifest.csv`
- `exp351_exp306_l1_full_convergence_audit_full_input.csv.gz`
- `exp351_exp306_l1_full_convergence_audit_full_output.csv.gz`
- `exp351_exp306_l1_full_convergence_audit_full_solver_status.csv.gz`
- `exp351_exp306_l1_full_convergence_audit_cross_run_parity.json`
- `exp351_exp306_l1_full_convergence_audit_full_gate.json`
- `exp351_exp306_l1_full_convergence_audit_summary.json`

各CSVはrows、schema/content SHAを保存する。gzipはraw SHAとdecompressed SHAを分け、decompressed SHAを主証拠にする。truth、error target、formation、MRR/top3/RMSE、prediction、submissionを生成物へ含めない。

## 実装方針と実装状態

- exp306 compact self-contained trainを参照元にし、必要なraw preparation、L1 solver、validation、SHA utilityだけを抽出する。
- RTS、Stage 0 branch selection、科学score、prediction/submission logicを持ち込まない。
- Jupytext percent形式の別名compact train候補を先に作り、canonical Notebookは採用承認までscaffoldのままにする。
- inference候補は常にfail-closeし、sample submissionのcopyも行わない。
- synthetic testでparent anchor guard、coverage、1-series failure、parity mutation、runtime超過、forbidden columnを検出する。
- 2026-07-23の実装依頼により、上記の別名compact train/inference候補と11件のcontract testsを実装した。
- 正規Notebookはscaffoldのまま維持し、Kaggle package/push/run flagもfalseのままとした。

## 再現性設計

- seed policy: RNGなし、well IDsはstable sort。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: solverはsingle worker、BLAS threads 1。結果はcanonical sort後に保存する。
- CPU/GPU: Kaggle CPU、GPU/TPU/internet off。
- input/output/status: schema/content SHA、raw gzip SHA、decompressed SHA、row/well/series countを保存する。
- parent kernel source: kernel idに加えてversion/id_noとartifact SHAを検証する。
- model/prediction/submission SHA: 非該当。
- deterministic anchor: submission anchorではない。固定inputと環境に対するsolver technical reproducibilityだけを主張する。
- Kaggle bootstrap: 実装・push承認後に正のconfig/sourceから再生成し、bootstrap内のparent anchor、L1設定、run flag、CPU/internet/thread設定を照合する。

## リスク

- リークリスク: horizontal TVTやtruth/scientific scoreを読むとtechnical-only契約が壊れる。load時点のschema allowlistとsummary null guardで停止する。
- 数値再現性リスク: NumPy/SciPy/BLAS差でiterationやraw float bytesが変わり得る。親64-well exact SHAをhard gateにし、差分を丸めで救済しない。
- ランタイム/メモリリスク: Stage 0外挿は約304秒だが、全well長分布やI/Oで増える可能性がある。wall time 8.5時間をhard gateにし、single runとする。
- 科学的リスク: 全件収束してもL1 denoiserが有用とは限らない。PASSは別科学実験の設計資格だけで、exp304 selected SWTを変更しない。
- 運用リスク: parent kernel sourceのlatest versionが変わる可能性がある。version 1 artifact SHAが一致しない限りsolverを開始しない。
- scope creep: RTS rescue、L1 iteration/tolerance/lambda/rho grid、scientific score、HMM/PF/Beam、inference、submissionを本実験へ追加しない。

## 次のアクション

Kaggle CPU version 1は9 horizontal series未収束で固定AND gateをFAILした。設計済みfailure policyに従ってtechnical negativeとして閉じ、solver parameter救済、scientific score、inference、submissionへ進めない。
