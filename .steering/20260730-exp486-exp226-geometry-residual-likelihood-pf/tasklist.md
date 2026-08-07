# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- raw-test inferenceとsubmissionは未承認。

## 完了

- exp279/281/419/404/417の差分と結果を確認した。
- 二つの独立PF variant、列allowlist、実行量、gate、再現性を確定した。
- backlog、steering、design-only scaffoldを作成した。
- compact self-contained Stage 0候補とfail-closed inference guardの実装承認を得た。
- Jupytext percent形式のcompact self-contained train候補を実装した。
- absolute unaryとslow residual-offsetのNumba PF kernelを実装した。
- exp226 allowlist、fixed32 SHA、common stable seed、両variant freeze後の
  truth/control/role/fold attachを実装した。
- target-free geometry / offset / ESS / resampling mechanism ledgerと、
  input/scientific/runtime/prediction content SHAを実装した。
- Stage 0の2 variants、64 PF well-runs、8,192 seed-well、4,096,000
  particle starts、control/model/HMM/Beam/GPU rerun 0をconfigへ固定した。
- fail-closed inference guardを実装した。
- 専用test 14件、Jupytext roundtrip、`py_compile`、ruff
  F821/F401/E9、strict experiment / project / config validationをPASSした。
- repository full test 1,983 nodeidを`make test` exit 0でPASSした。
- 正規train/inference Notebook placeholderを上書きしていないことを確認した。
- 正規train Notebook採用、canonical Kaggle package / push、fixed32
  Stage 0実行の承認を得た。
- canonical Kaggle private CPU kernel version 1（id_no `129170320`）を
  pushし、同一IDを監視して`COMPLETE`を確認した。
- 2 variants ×32 wells、8,192 seed-well、4,096,000 particle startsを完走した。
- 事前固定runtime投影`180,871.020 sec > 30,600 sec`とstrict residual
  support boundをFAILし、`stage0_fail_closed`でbranchを閉じた。
- fixed32記述RMSE、truth-late ledger、runtime、gate、prediction /
  mechanism artifact SHAを記録した。
- test-side geometry再生成、Stage 1、inference、submission、gate/parameter
  救済を行わないと確定した。
- その後、ユーザーがruntimeを許容してStage 1進行を明示承認した。
- Stage 0 original FAILを保持し、supportは最大約`1.1e-15`の丸め誤差として
  `1e-12` toleranceで監査するStage 1例外契約を確定した。
- 全773 wellsの二variant生成、SHA freeze、truth-late CV、variant別gateを
  実装し、専用test 16件、Jupytext、構文、ruffをPASSした。
- Stage 1 version 2で98,944,000 particle startsと両variant freezeを完了した。
- truth-late exp209 SHAの62文字manifest typoによるERRORを診断した。
- freeze済みprediction / mechanism ledger / auditをSHA検証し、
  private Dataset `kentookumura/exp486-v2-stage1-frozen-targetfree`へ回収した。
- current PF rerun 0のversion 3 resume loaderを実装し、専用test 17件、
  Jupytext、構文、ruff、strict exp validationをPASSした。
- version 3の展開済みledger再serialize SHA差を診断し、payload SHAと
  schema/coverage/finite値を分離検証するversion 4へ修正した。
- canonical Kaggle version 4を完了し、technical gate全項目PASSを確認した。
- absolute `9.726938029`、residual `11.139812021`、保存control
  `10.914522073`を記録した。
- absoluteはby-well p95/worst、residualはpooled/fold/scope/tailで科学gateを
  FAILし、eligible variant 0でterminal closeとした。
- result、metrics、README、SESSION_NOTES、experiment summary、
  KAGGLE_DIRECTIONへ最終結果とSHAを記録し、専用test 18件をPASSした。
