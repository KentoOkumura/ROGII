# 要件

## 依頼

物理モデル単体で Public LB 6.5 を目指す研究系列の第1段階として、current-testで再生成可能な
純物理candidate bankの到達可能性をrow / block / whole-well oracleで監査する。

初回設計段階ではbacklog、steering、実験scaffoldと設計契約だけを確定した。2026-07-19の追加承認で
compact実装、続く実行承認でcanonical採用とKaggle CPU auditまで完了した。raw-test inferenceと
submissionは引き続き作成・実行しない。

後続の第2段階`prefix_calibrated_latent_registration_gr_evidence`、第3段階
`joint_physics_candidate_registration_semimarkov_smoother`、条件付き第4段階
`mode_loss_triggered_candidate_birth_beam`について、開始条件、固定内容、禁止事項を同時に記録し、
別セッションが独自に目的や分岐を変更できないようにする。

## 制約

- Route: `pf_beam`。
- 最終目的は物理モデル単体のPublic LB 6.5。ML selector、LightGBM/CatBoost/XGBoost、ML予測とのblendは使わない。
- 第1段階のprimary bankはexp263 Stage 1でcurrent-test再生成済みの12候補に固定する。
  - 6 primitive: `exp226_k16`, `selfgr_hmm_a070`, `likpf_mean`, `exact_hmm`, `pf_ancc`, `beam_mean`
  - 5 raw-test pair: exp263で固定済みの50/50 pair
  - 1 fixed formula: `exp226_w500_50_50`
- `geop_hmm`、exp268、exp270、exp271、exp291、exp292など後発候補はprimary判定へ混ぜない。
  exp293完了後のcandidate-birth候補または補助証拠としてだけ参照する。
- candidate値、formula、fold、row identity、evaluation suffixを変更しない。
- candidate生成、candidate選択、GR score、補正、平均、softmax、decoder、学習を行わない。
- true TVTはcandidate tableとbank manifestをSHA freezeした後のoracle readoutだけに使う。
- row oracle、非重複H128/H256/H512 block oracle、whole-well oracleを同じbankで計算する。
- primary判定粒度はH512とする。末尾short blockは捨てず、同じwellの最後のblockとして評価する。
- anchorは`exp226_w500_50_50` OOF RMSE `8.2383315465`、目標値は`6.5`に固定する。
- 再現性は`docs/06_reproducibility.md`に従う。gzipはraw SHAに加えてdecompressed content SHAを主証拠にする。
- 第1段階は1 audit contract、LightGBM config 0、trained fold 0、booster 0、HMM/PF再生成0、GPU 0。
- compact実装、canonical採用、Kaggle CPU version 2完了済み。1回のpush承認は消費済み。

## 受け入れ基準

- steering 3文書、実験`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`に未記入placeholderが残っていない。
- primary 12候補のidentity、formula、順序、評価粒度、oracle定義、support PASS/FAIL条件が明記されている。
- `downstream_branch_contract.md`に第2/3/4段階の開始条件、固定内容、成功条件、禁止事項が記録されている。
- support PASSなら第2段階、support FAILなら第4段階へ進み、第2段階FAILから第4段階へ自動分岐しないことが明記されている。
- 第3段階は第2段階PASS時だけ開始できる。
- deterministic anchorとは扱わず、入力・bank・oracle readoutのcontent SHAを実行時に記録する設計になっている。
- `make validate-exp EXP=exp293_physics_only_candidate_bank_headroom_contract`が通る。
- Jupytext train候補とfail-closed inference候補、専用contract testsが実装されている。
- H512 pooled/fold/support判定とSHA evidenceが記録され、固定分岐へ進める。
- raw-test inference、submission、oracle/selected row predictionは未生成のままである。

## 次

support PASSのためStage 2だけを次候補とする。Stage 4は開始せず、Stage 2実装は別承認を待つ。
