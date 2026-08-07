# exp488_isolated_gr_shock_prior_hold_support_only セッションノート

## 目的

exp482のzero-shock well対照群を不要とするユーザー指示を、exp482の履歴を
書き換えず独立したsupport-only分岐として検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage0_failed_closed`
- 親: `exp482_isolated_gr_shock_prior_hold`
- scientific parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV / LB: なし / なし
- Stage 0 run: Kaggle version 2で完了
- Stage 1 / inference / submission: 実施しない

## exp482からの差分

- 変更: target-free manifestをsupport32 + zero-shock control32から
  support top32だけへ変更。
- 不変: raw shock、message agreement、current-emission conflict、
  row-local LOO readout、HMM state/grid/transition/emission/prior、saved parent、
  support ordering、科学gate。
- zero-shock controlがないため、非発火wellの群間安全性は主張しない。
- support内のby-well p95 / worst-well gateとrow-local実装testは維持する。

## 実行量契約

- scientific candidate: 1。
- raw-only census: 773 wells / 3,783,989 suffix rows。
- unchanged exp209 parent-message HMM replay: 32 wells。
- candidate state-modifying HMM: 0。
- saved exp209 prediction rerun: 0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- PF / Beam / GPU: `0 / 0 / 0`。
- Stage 1最大773 HMM replayは別承認。

## 承認

- 2026-07-30、ユーザーの「安全に性能検証するための事前固定した対照群は
  いらないので次に進む」を、exp488の設計、実装、canonical train Notebook採用、
  Kaggle package、support-only Stage A0/A1 runの承認として記録。
- Stage 1、inference、submissionは承認範囲外。

## 再現性

- `docs/06_reproducibility.md`: 確認済み。
- RNGなし、single worker / single numba thread、CPU-only、internet disabled。
- freeze順: raw census → support32 manifest → message → trigger → prediction →
  SHA readback → truth/fold/error join。
- gzipはdecompressed content SHAを主証拠にする。
- 初回成功runをdeterministic anchorにしない。
- canonical train Notebook SHA256:
  v1 `f0f03dd21708373886d06e25538c5544de219c872ff2f74cd9e8dd8019fa4bee`;
  v2-fix `b7675f28dfb98050caa1e746137002edb778c5b0166790f25b15e1aa210f9ca5`
- fail-closed canonical inference Notebook SHA256:
  `b1a78348c6d368ec7c12e1ff85a1965349b6f4ec361926aec259658fe11eeff8`
- 親exp482 compact trainは2,269行・10章、exp488は2,216行・10章。
  control matchingとcontrol metric/gateだけを除き、HMM、trigger、truth-late、
  artifact orchestrationをNotebookセル上に維持した。
- `__file__`依存はtrain/inference sourceとも0。
- strict package:
  `experiments/exp488_isolated_gr_shock_prior_hold_support_only/kaggle/train`
- initial long-slug package train / metadata SHA256:
  `49c413468b91ed9646816ba09bd84dfdbb249aaebe32120ba602af46493c953b` /
  `b3c440db63696c23ae9d1562c70e63fd1a79620566e6df14d41da540a64ddc07`
- metadataはprivate、CPU、internet disabled、run-on-push、
  exp209/exp226 kernel sourceを確認済み。
- 埋め込みconfigはsource configと一致し、support32 / control0 /
  HMM replay32 / Stage 1・inference・submission disabledを確認済み。
- push直前の専用pytestは`14 passed`、exp408/440/482/488回帰は
  `51 passed`。Ruff F821/E9、format、Jupytext、strict validationはPASS。
- 初回planned slug
  `exp488-isolated-gr-shock-prior-hold-support-only-train`は54文字で、
  title-derived slugとの一致確認後もKaggle `SaveKernel 400`。
- 同slugのpullは403で、Kaggle側に利用可能なNotebookは作成されていない。
- 同じexp488のまま、意味を保った34文字のcanonical slug
  `exp488-gr-shock-support-only-train`と同じtitle由来slugへ短縮して再packageする。
- current packaged train Notebook SHA256:
  `13a23c72bd6e10226d9d457ba009960e1a48ede43debf7a4620865099506f0fb`
- current kernel metadata SHA256:
  `195fba678c6161febac7424e26c65f731c08a47247e41a754d9bf873a9b8b804`
- `2026-07-30 12:32:09 UTC`: version 1をpush。
- Kaggle `id_no=129170127`、push直後statusは
  `KernelWorkerStatus.RUNNING`。
- pull metadataでprivate、CPU、internet disabled、exp209/exp226 sourceを再確認。

## Kaggle version 1 technical ERROR

- `2026-07-30 13:06:17 UTC`: `KernelWorkerStatus.ERROR`。
- log elapsed: `2032.356755164 sec`。
- 32-well HMM、prediction、truth-late metrics、gate計算後、
  `stage0_gates.json`保存時に停止。
- error:
  `TypeError: Object of type bool is not JSON serializable`。
- 原因: `to_jsonable`が`np.integer` / `np.floating`を処理する一方、
  pandas由来の`numpy.bool_` gate値を標準`bool`へ変換していなかった。
- 科学条件、manifest、trigger、HMM、gateは変更しない。
- 修正は`np.bool_ -> bool`のgeneric JSON conversion 1件だけ。
- 専用serialization testを追加し、同じcanonical kernelのversion 2で再実行する。
- v2-fix検証: 専用pytest`15 passed`、exp408/440/482/488回帰
  `52 passed`、構文、Ruff、format、Jupytext、strict validation PASS。
- v2 packaged train Notebook SHA256:
  `0f8e904081f12c260b0843bb555052ba36743e624c9bb38976e893e6bc60a360`
- v2 package configはsource configと一致し、kernel metadata SHAは
  `195fba678c6161febac7424e26c65f731c08a47247e41a754d9bf873a9b8b804`。
- `2026-07-30 13:09:47 UTC`: 同じcanonical kernelのversion 2をpush。
- push直後status: `KernelWorkerStatus.RUNNING`。

## Kaggle version 2結果

- `2026-07-30 13:42:12 UTC`: `KernelWorkerStatus.COMPLETE`。
- notebook log elapsed: `1922.239474176 sec`。
- status: `stage0_failed_closed`。
- stage: `stage_a0_a1_support32`。
- decision:
  `stage0_failed_close_without_trigger_threshold_or_output_rescue`。
- eligibilityはPASS:
  - raw census: 773 wells。
  - isolated raw-shock: 17,047 rows。
  - support: 763 wells。
  - zero-shock: 10 wells。今回はreport-onlyでgateには使わない。
- support32:
  - 32 wells / 183,093 rows。
  - control well: 0。
  - final trigger: 0 rows / 0 wells / 0 folds、fraction 0.0。
  - unchanged parent-message HMM replay: 32。
  - candidate state HMM / saved parent rerun / LightGBM config /
    trained fold / booster / fitted model / PF / Beam / GPU: 全て0。
- score:
  - saved parent RMSE: `7.668975974817114 ft`。
  - candidate RMSE: `7.668975974817114 ft`。
  - improvement: `0.0 ft`。
  - improving folds: `0 / 5`。
  - by-well delta p95 / worst: `0.0 / 0.0 ft`。
  - fold 0: 53,966 rows / 10 wells / `8.627291 -> 8.627291 ft`。
  - fold 1: 32,695 rows / 5 wells / `5.770497 -> 5.770497 ft`。
  - fold 2: 17,921 rows / 3 wells / `3.308634 -> 3.308634 ft`。
  - fold 3: 42,595 rows / 8 wells / `6.237724 -> 6.237724 ft`。
  - fold 4: 35,916 rows / 6 wells / `10.298702 -> 10.298702 ft`。
- technical:
  - stage0 elapsed: `1899.470163996 sec`。
  - full runtime projection:
    `39059.748071987335 > 30600 sec`でFAIL。
  - peak RSS: `1.4179039001464844 GB`。
  - truth-late: 183,093 rows / 32 wells、freeze前forbidden read 0。
  - support-only manifest、well overlap、finite、normalization、RSS、
    trigger fraction上限はPASS。
  - trigger最小row / well / fold、runtime投影、
    saved-parent replay parityはFAIL。
  - parent replay parityの実測差は最終log summaryに表示されていないため、
    bool判定だけを記録する。
- scientific:
  - triggerが0件なのでtrigger-row改善率とSSE削減は評価不能。
  - pooled改善0 ft、改善fold 0/5でFAIL。
  - candidateとparentが同一なのでtail delta 0だけはPASS。
- runtime versions:
  Python `3.12.13`、pandas `2.3.3`、NumPy `2.0.2`、
  Numba `0.60.0`。

## version 2生成物SHA

- scientific contract:
  `3f38c95eefa949561aaaf663d737f9b33cc60b94cb8fdc871916e7568a623508`
- raw census:
  `fdbb653e13bdd6132ffbe08d129fc44a744ed72b81fdc4d41ae04aa0848202cb`
- raw shock rows decompressed / gzip:
  `1615aa3504eba71a90dc5c36f782ba79ea34162f3bd2876043b132f703332116` /
  `b8236c4d31ffb5c4659bf091f851b5badab9d048e8358a25fa30877658ac40e2`
- support32 manifest:
  `52e50a44f8976aa58899971b603fd1c6675bd410ab0c3a681d4acd73dc3eb41d`
- message bundle:
  `ba00a800b0c53c3e3e7d5eac80ba0c5973f42b106177d681e322d74997d48ce2`
- diagnostics decompressed / gzip:
  `ba7c625e266067d03e725c51da9b60ca8cb4bb4065684723ff8fd9bd4e1d9207` /
  `81cd7f75355e06affbb6af4e604a3a65c86bac4eb3aa67f5c4cd1cc875ccc4f6`
- trigger / gzip:
  `9e92bdf46a0a63813f315dbd73ff8258ee2a94da4ab555bc3d18ec4326c7dad0` /
  `652aa9a5e78610f14aa907169e1678dd299403e6059cf93876f72abd8382af80`
- prediction / gzip:
  `10c6491e93c56228d0a346f6f05619016abba946134fde64065aae9f2391c17c` /
  `2d9a2dd56fcd5849dcf98cdbaf5c43ade98878ada250593af04f0fa12279296d`
- by-well / well audit / scope metrics:
  `b70ac5b04488db553e783dddbd52346452a1fcf70e615058a3350c5742de0de0` /
  `7741a90a75029c8b1fa433d4d84d2f0bcf05cb9ee72d1f575a4843bee9132296` /
  `54dbfbb82837c6295783ff45f6eb243ddea022b4b51490d8ecdae195f2fea836`
- gate / summary:
  `40e3d5eb3883cb73141eb55d06eefea11e514a82ec977fb51e2104b4ae07ed92` /
  `6a84c175e4386e875ba80c25673e8956fe5606a768d2eaa1a9a7f8d04f14ea3d`
- parent input decompressed:
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- fold input decompressed:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`

## 終了判断

- zero-shock対照群を外してshock集中sampleだけを評価しても、固定AND triggerは
  1行も発火しなかった。対照群不足ではなく、現条件で機構が非活性である。
- support32はCV / promotion evidenceではないが、全OOFへ進む機構根拠もない。
- 事前契約どおりthreshold、window、output、trigger条件のsame-data救済はしない。
- Stage 1、inference、submissionは行わずbranchを閉じる。
- Kaggle logに必要なgate、metrics、runtime、SHAが揃ったため、
  output archiveは取得していない。
