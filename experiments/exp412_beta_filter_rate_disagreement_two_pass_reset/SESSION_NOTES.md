# exp412_beta_filter_rate_disagreement_two_pass_reset セッションノート

## 目的

future betaとforward filterのrate disagreementをfirst passでfreezeし、second passの
rate transitionだけを方向付きde-stickする高リスク第二案を実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU Stage 0 Version 3完了 / `stage0_fail_closed`
- 優先度: P3・高リスク
- CV / LB: Stage 0 mechanism結果あり / LBなし
- prerequisite: exp411
- inference / submission: 無効
- 実装承認: 2026-07-28のユーザー指示
- Kaggle実行承認: 2026-07-28のユーザー指示「実行してください」

## 2026-07-26 設計確定

ユーザー依頼により、第一案exp411とは別の仮説・実行量を持つ第二案としてexp412を採番した。

### 根拠

- exp408 exclusive backward smoothing reversal SSE: `23.0444%`
- beta hurts truth strong SSE: `66.9676%`
- betaでtruth-rate massが回復しposition massが悪化するSSE: `38.3313%`
- saved ledgerの`|smoothed-filtered rate| / max(filtered_std, 0.005)`:
  median`0.556`、p90`1.548`、p95`1.991`、p99`3.183`

### 固定した変更

- beta-filter standardized gap threshold: `2.0`
- rolling window: 16 rows
- qualifying rows: 8以上
- sign consistency: 75%以上
- second-pass stay mass transfer: 10%
- active scheduleはfirst passでfreezeし、treatmentから再計算しない
- edge: outward neighborがないsource stateだけno-op

### 実行量

Stage 0予定:

- baseline / treatment variants: `1 / 1`
- HMM well-runs: `32 + 32 = 64`
- parent control regeneration: あり
- model / LightGBM config / trained fold / booster / PF / Beam / GPU: 0

Stage 1予定:

- baseline / treatment variants: `1 / 1`
- HMM well-runs: `773 + 773 = 1,546`
- reporting folds: 5
- parent control regeneration: あり
- model / booster / PF / Beam / GPU: 0

parent internal messageが全wellで未保存のため再生成が必要だが、実行は未承認。

## exp411との順序

- exp411がtrigger support / future evidence不足でFAILした場合だけ実装資格を得る。
- exp411がpromotion PASSならexp412は未実装のままcloseする。
- ユーザーoverrideがあれば順序を変更できる。

## 2026-07-28 exp411先行条件の判定

exp411 Stage 0 Version 5はtechnical gate 13 / 13 PASS後、mechanism gate 2 / 6 PASSで
`stage0_fail_closed`となった。

- future-rate direction agreement: `0.225397 < 0.60`
- passing folds: `0 / 5 < 4 / 5`
- control active-row fraction: `0.136119 > 0.10`
- persistent minus control active-well fraction: `0.0 < 0.20`

この結果は「causal trigger support / future evidence不足」という事前登録済みの
exp412実装資格を満たした。

## 2026-07-28 Stage 0実装

ユーザーの「exp412を実装してください」を実装承認として扱い、Kaggle実行とは分離して
compact self-contained候補を実装した。

### 固定manifest

- roles: backward cause 8 / forward cause 8 / matched control 16
- unique wells: 32
- backward / forwardの両roleが5 foldsを含む
- control pool: exp408 persistent scope外の323 wells
- control match: fold、suffix rows quartile、raw-GR missing quartile、
  prefix rows quartile
- maximum control quartile distance: 1
- manifest SHA256:
  `1edb1e1481af84af4e8178fb6e0743fa40315eab0b7441eeff9232b571f93c30`
- exp408 episode summary SHA256:
  `b230ffc759e6ee4891f22809b3f3c8a8796681fb461ec0b7215b94a352bf0ab0`

cause membershipはsample選択と全prediction freeze後のreadoutにだけ使い、HMM interfaceへ
渡さない。

### two-pass実装

- first pass: frozen directionを全0にしたexp209 exact HMM
- first-pass readout: filtered rate mean / std、smoothed rate mean
- trigger:
  `z_beta=(smoothed-filtered)/max(filtered_std,0.005)`、inclusive 16-row
  window、qualifying 8 rows以上、majority sign 75%以上
- second pass: frozen active rowへ入るtransitionだけstay mass 10%を隣接rate stateへ移す
- edge outward state: no-op
- second-pass posteriorからscheduleを再計算しない
- baselineがsaved exp209 predictionと`1e-5 ft`以内で一致しないwellは、
  そのwellのtreatment前にfail-close
- 32 wellsのbaseline message / schedule / baseline / treatment predictionをfreeze後に
  TVT truthとexp408 cause intervalをlate join

### 実行contract

- active treatment variant: 1
- Stage 0 baseline / treatment / total HMM well-runs: `32 / 32 / 64`
- parent control reruns: 32
- LightGBM config / trained fold / booster / model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- `stage_0_execution_approved: false`
- `kaggle_execution_authorized: false`

### 生成した実装

- `exp412_beta_filter_rate_disagreement_two_pass_reset_compact_selfcontained_train.py`
- 同Jupytext変換Notebook
- fail-closed compact self-contained inference候補
- `build_stage0_manifest.py`
- fixed32 manifest / metadata
- 専用contract test

既存の正規`*_train.ipynb` / `*_inference.ipynb`はplaceholderのまま上書きしていない。
正規Notebook採用とKaggle package作成は別承認の対象とする。

### 静的検証

```text
dedicated pytest: 12 passed
dedicated + notebook contract pytest: 16 passed
py_compile: PASS
Ruff F821: PASS
Jupytext train/inference round-trip: PASS
make validate-exp (strict): PASS
```

科学的親exp209にはcompact self-contained版がないため、直近の同kernel実装参照である
exp411 compact trainと比較した。exp411は2,255行、exp412は2,312行。exp412は
path/SHA、fixed32 input、exp209 preprocessing、two-pass kernel、schedule freeze、
truth/cause late join、gate、orchestration、metrics保存の9章を持ち、exp411の役割slotを
欠かしていない。

Notebookの初回実行はローカルでもKaggleでも行っていない。

## 再現性メモ

- `docs/06_reproducibility.md`確認済み。
- RNGなし。well / row / pass順を固定する。
- first-pass parityをsaved exp209へ`1e-5 ft`以内で必須とする。
- sample、first-pass message、active schedule、baseline / treatment prediction、
  metricsのdecompressed content SHAを保存する。
- deterministic submission anchorではなく、submissionは生成しない。
- push前にloose / bootstrap config、Notebook body、asset SHAを照合する。

## 未実施

- 正規Notebook採用
- Kaggle package / push / run
- inference / submission

## 次のアクション

正規Notebook採用の判断後も、parent control再生成を含むStage 0
`32 baseline + 32 treatment = 64 HMM well-runs`は、別の明示承認を得るまで
package / push / runしない。

## 2026-07-28 Stage 0実行承認

ユーザーの「実行してください」を、正規Notebook採用とKaggle private CPU Stage 0の
package / push / run承認として記録した。

- active treatment variant: 1
- baseline / treatment / total HMM well-runs: `32 / 32 / 64`
- parent control HMM reruns: 32
- LightGBM config / trained fold / booster / model: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- Stage 1 / inference / submission: 未承認

`stage_0_execution_approved`と`kaggle_execution_authorized`をtrueへ変更し、
compact trainとfail-closed inference候補を正規Notebookへ採用する。

### push前package監査

- 正規train Notebook SHA256:
  `0c2efa5fe6cc35868cc6841dc49f0856db25b4c0f1e56f7a07ce69193b9db1c7`
- Kaggle package Notebook SHA256:
  `af872dc63d8c9451a397f258d56c0a768713f78e9de385609a6352908f164c2e`
- bootstrap ZIP SHA256:
  `aa5f8774d623b5bd554e8c24df0303938f430183b6ca3eda96ff545644b87294`
- bootstrap manifest: 34 / 34 entriesのbyte数とSHA256を照合
- loose / bootstrap / repository `config.yaml`: byte一致
- fixed32 manifest、manifest metadata、exp408 episode summary: bootstrapとrepositoryでbyte一致
- 最初のpushはKaggle APIの`SaveKernel 400`で拒否され、実行は開始しなかった。
  原因は当初のslug/titleが57文字でKaggleの50文字上限を超えたこと。
- 48文字へ短縮したcanonical kernel id:
  `kentookumura/exp412-beta-filter-rate-gap-two-pass-reset-train`
- 48文字へ短縮したcanonical title:
  `exp412 beta filter rate gap two pass reset train`
- private / CPU / internet off / run-on-push: 確認済み
- kernel source:
  `kentookumura/exp209-joint-exact-parity-train`
- push前検証: dedicated + notebook contract pytest 18件、F821、
  Jupytext round-trip、strict experiment validationすべてPASS

### Kaggle Stage 0 version 1

- push: 成功
- 最終status: `ERROR`
- kernel id no: `128917257`
- Kaggle pull後のid/title/source: canonical metadataと一致
- pull後Notebookと送信packageの全21 cell連結source SHA256:
  `0d357cd75a97f1d74bbe070c22bf8181523065db4b59cd42eb09abf7e2acaba7`
- 実行先:
  `https://www.kaggle.com/code/kentookumura/exp412-beta-filter-rate-gap-two-pass-reset-train`

Version 1は開始約22秒でsaved exp209 control SHA guardに停止した。設定へ転記したSHAが
62文字で、Kaggle artifactおよびexp411の確定済み64文字SHA
`8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
から`8b`が欠けていた。HMM well-runは0件、科学resultは未生成。2つのconfig fieldを
既知の64文字SHAへ訂正し、完全一致を要求する回帰testを追加した。訂正後は17 tests、
strict validation、34 / 34 bootstrap manifest照合がPASSした。Version 2の実行量は
当初契約どおりbaseline 32 + treatment 32 = 64 HMM well-runsで、Version 1との累計も
64のまま。

Version 2は同じcanonical kernelへpush成功し、`RUNNING`を確認した。

Version 2は最初のwell `14fee784`のbaseline 1 HMM well-runを完了後、保存exp209との
float64差`0.000958307548 ft`で`1e-5 ft` guardに停止した。科学resultとtreatmentは
未生成。元exp209 kernelを現行Numba 0.60.0 / Numpy 2.0.2で独立再実行するとexp412と
全4910行・log-likelihoodが完全一致した。exp209保存処理の`_numeric_frame`が予測を
float32へ変換するため、約12,000 ftで生じる量子化差だった。

guard閾値は変更せず、再計算値と保存値をexp209の保存表現float32へ正規化してから
比較するよう修正した。fresh JITで18 tests PASS。`14fee784`実wellの正規化前差は
`0.000958307548 ft`、正規化後差は`0.0 ft`で、1 float32 ULPの差は従来どおり
`1e-5 ft`を超えてfail-closeする回帰testも追加した。

Version 3の予定実行量はbaseline 32 + treatment 32 = 64 HMM well-runs。
Version 1の0件、Version 2のbaseline 1件を含む累計は65 HMM well-runs。

Version 3はpush成功し、`RUNNING`を確認した。

## 2026-07-28 Stage 0 Version 3完了

canonical private CPU Version 3（kernel id no `128917257`）を完了した。

- baseline / treatment / total HMM well-runs: `32 / 32 / 64`
- elapsed: `2,142.435153秒`
- peak RSS: `0.991058 GB`
- full runtime projection: `51,753.199176秒`
- version 1 / 2 retryを含む累計HMM well-runs: `0 + 1 + 64 = 65`
- model / LightGBM config / booster / PF / Beam / GPU: 0
- scientific contract SHA:
  `9f216b213907c42009ce1a7dbcb48f8dc27d7e8861696df3d4fb5432a7da3993`

32 / 32 wellsで保存exp209のfloat32 storage parityは最大差`0.0 ft`。
finite coverage `1.0`、maximum normalization error `2.438425e-08`、
zero-schedule self parityも差`0.0 ft`だった。

technical gateは12 / 13 PASS:

- active rows / fraction: `5,902 / 0.0387517`、PASS
- active wells: `21 / 32`、PASS
- peak RSS: PASS
- full runtime projection:
  `51,753.199176 > 30,600秒`、FAIL

mechanism gateは5 / 6 PASS:

- beta direction correction agreement:
  `0.776347 >= 0.60`、PASS
- fold agreement:
  `0.997364 / 0.834123 / 0.726087 / 0.366008 / 0.874887`、
  strict PASS `4 / 5`、PASS
- backward active-row coverage:
  `0.093159 >= 0.05`、PASS
- backward cause SSE reduction:
  `-0.069575 < 0.10`、FAIL
- forward cause SSE regression:
  `+0.013257 <= 0.02`、PASS
- control RMSE delta / active-row fraction:
  `+0.005836 ft / 0.029605`、PASS

backward causeの改善は`fae0c593 -1.856824 ft`、
`57f05c51 -3.080338 ft`に集中し、`a9c9b150 +3.160318 ft`、
`c9e980e8 +2.260255 ft`などの悪化を相殺できなかった。方向自体は多くのactive rowで
正しいが、固定10%のstay-mass transferはwrong position basinの修復へ安定して
つながらない。

実artifactを
`/tmp/kaggle-output-exp412-v3`へ限定取得して監査した。

- activation schedule: 152,303 rows、raw
  `b5064975...8fcdaa`、decompressed `18cdb7da...7d85b`
- predictions: 152,303 rows、raw
  `2852a250...728fc`、decompressed `29f3148c...fccd6`
- well metrics: 32 rows、`28fbddbf...a1d82`
- direction truth-late: 5,902 rows、`5e4bab61...d3a958`
- cause episodes: 20 rows、`b2bbcf21...d27519`
- summary: 6,997 bytes、`621decde...a167379`
- input manifest: 1,414 bytes、`bfcd1852...0c8e1b5`

Kaggle outputのmetrics / summaryとlocal記録のresult / gateを一致させ、全CSV row数、
raw SHA、gzip decompressed SHAを照合した。

結論は`stage0_fail_closed`、`promotion_eligible: false`。同じOOFでthreshold、
window、sign fraction、transfer量、well/row gateを探索しない。Stage 1 full、
inference、submissionは行わずbranchを閉じる。次のHMM候補は既存backlogの
独立単一因子`exp424_exp209_momentum1_exact_hmm_ablation`を維持するが、
exp412失敗を理由に優先度を上げずP3のままとする。
