# exp458_acceleration_state_exact_runtime_engine_audit

## 状態

- ルート: PF/Beam
- 状態: Stage 0A数値parity FAIL・terminal close
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-30
- 科学仕様の構造参照: `exp444_acceleration_state_exact_hmm`

## 仮説

exp444の3状態acceleration exact HMMは、状態、遷移、emission、posterior readoutを
一切変更しなくても、scaled probability-spaceの因子化/fused計算、
exact-bit `delta_MD` cache、4-well外側並列によりKaggle CPUの固定runtime上限内で
同じ結果を再現できる。

これはexp444の科学結果を救済する仮説ではなく、同じ数理モデルを高速に計算できるか
だけを問う独立runtime仮説である。

## 実装

- log-space edge単位logsumexpをfloat64 scaled probability-spaceへ置換した。
- acceleration 3x3、rate 41x41、position 5-offsetを因子化/fusionした。
- 同一well内のbitwise同一`delta_MD`だけOU kernelを再利用する。
- outer workers 4、worker内Numba/BLAS thread 1、stable output orderを実装した。
- exp444のscientific contract SHAと全state supportは変更しない。
- 保存exp444 fixed4はcandidate 2 repeatsのfreeze後だけloadし、prediction/std、
  acceleration posterior、rate diagnosticを固定許容差で比較する。

## 検証方針

- Stage 0A: 保存exp444 fixed4をload-only比較し、candidate fixed4を2回実行。
- 数値: prediction、std、acceleration posterior、rate diagnostic、dense reference、
  normalization、kernel parityをAND gateで判定。
- runtime: 2回の遅い方で`>=4.75x`、fixed32/full投影`<=3,600/30,600 sec`。
- memory: peak RSS `<=25 GB`。
- Leakage Check: truth/role/fold/episode/causeはfreeze前read 0。
- Stage 0B/1、inference、submissionは各前提PASS後も別承認。

## 実行入口

- 学習 notebook: `exp458_acceleration_state_exact_runtime_engine_audit_train.ipynb`
- 推論 notebook: `exp458_acceleration_state_exact_runtime_engine_audit_inference.ipynb`
- compact train候補を正規train Notebookへ採用し、Stage 0Aの実行入口とする。
- 実装候補:
  `exp458_acceleration_state_exact_runtime_engine_audit_compact_selfcontained_train.py`
  / `.ipynb`。
- inference候補はfail-closed guardだけを実装した。
- inferenceの正規Notebookはtemplate scaffoldのままで、実行しない。

## 結果

Kaggle private CPU version 2を完了し、Stage 0Aは
`stage0a_fail_closed`となった。runtimeは遅いrepeat `72.755703 sec`、
exp444比`10.258353x`、fixed32/full投影`582.045625 / 14,114.606407 sec`、
peak RSS `13.033188 GiB`で全PASSした。2 repeatsのprediction/posterior/
diagnostic SHAも完全一致した。

一方、保存exp444に対する最大差はprediction mean `1.0413506e-4 ft`、
std `6.3565741e-5 ft`、acceleration posterior `8.9772640e-6`で、
固定閾値`1e-5 / 1e-5 / 1e-7`をFAILした。rate diagnostic
`1.0862974e-6 <= 5e-6`はPASSした。CV/LBはない。

## 所見

- exp444のruntime不足は科学仕様を変えずに検証できる独立の実装仮説へ切り出した。
- exp399の成功実績だけでなくruntime varianceも反映し、2 repeatsの遅い方を採る。
- small dense referenceはprediction最大誤差`1.82e-12 ft`、
  acceleration posterior最大誤差`3.33e-16`でPASSした。
- synthetic入力でexp444 log-space engineとの固定許容差parityをPASSした。
- runtime高速化は成立したが、長い実データtrellisでのexp444数値同値性は
  事前閾値を満たさなかった。
- small dense parityだけでは長系列の累積数値差を保証できないことが分かった。

## 次

exp458を再実行・救済せず閉じる。Stage 0B/1、inference、submissionへ進まない。
原因調査を行う場合も、保存済みv2/exp444だけを使う長系列first-divergence
studyとして別管理し、exp458のgate変更や再実行には使わない。

## リスク / 注意

- probability-space化は演算順が変わるため、親とのbitwise一致ではなく固定許容差で判定する。
- Kaggle CPU variance対策として2 repeatsの遅い方をruntime gateに使う。
- 4 processの実測RSSとthread oversubscriptionをfail closedで監査する。
- Stage 0A PASSは科学的改善の証拠ではない。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
