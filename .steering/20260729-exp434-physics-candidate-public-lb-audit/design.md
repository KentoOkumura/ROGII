# exp434 設計

## 結論

exp263 Stage 1でhidden-safeに再生成できる12本の物理候補を凍結し、
既存3件のPublic LBを再利用したうえで、未提出9件だけを2日・`5 + 4`件で
測るlate-stage LB censusとする。

これは新しい物理モデル実験でも、LBで係数を選ぶ実験でもない。OOF表へ
候補定義が一致するPublic LB列を追加し、OOF/LBの順位整合性を記述する
提出監査である。

## 実験範囲

- 対象実験:
  `exp434_physics_candidate_public_lb_audit`
- Route:
  `pf_beam`
- 親実験:
  `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 説明元:
  `exp264_exp263_candidate_confidence_dual_selector/physical_model_summary.md`
- 変更する変数:
  `selected_submission_candidate`だけ
- 固定する変数:
  6 primitive generator、5 pairの式、固定3-way式、dtype、seed、source、
  row order、sample schema、runtime、Kaggle inputs
- 作らないもの:
  train prediction、OOF、candidate、model、selector、feature、weight、
  postprocess、Private LB推定

## 候補契約

| ID | 種別 | 式 | OOF RMSE | LB状態 |
| --- | --- | --- | ---: | --- |
| `exp226_k16` | primitive | direct | 9.427110 | 既存9.837、同一性gate後に再利用 |
| `selfgr_hmm_a070` | primitive | direct | 11.349943 | batch 2 |
| `likpf_mean` | primitive | direct | 11.594898 | 既存9.721、同一性gate後に再利用 |
| `exact_hmm` | primitive | direct | 11.938287 | batch 2 |
| `pf_ancc` | primitive | direct | 14.493051 | batch 2 |
| `beam_mean` | primitive | direct | 15.774327 | batch 2 |
| `exp226_k16__selfgr_hmm_a070` | pair | 50:50 | 8.532715 | batch 1 |
| `exp226_k16__exact_hmm` | pair | 50:50 | 8.635074 | batch 1 |
| `exp226_k16__likpf_mean` | pair | 50:50 | 8.813822 | batch 1 |
| `selfgr_hmm_a070__likpf_mean` | pair | 50:50 | 10.123457 | batch 1 |
| `likpf_mean__exact_hmm` | pair | 50:50 | 10.269697 | batch 1 |
| `exp226_w500_50_50` | fixed | 50/25/25 | 8.238331 | 既存7.800、exact SHAで再利用 |

OOF値の正はexp263 Stage 0 / exp264 physical summaryとし、丸め前値が
利用可能な場合はmetricsへ保存する。表示は6桁に統一する。

## 既存LB再利用

### K16

- source experiment:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- kernel:
  `kentookumura/exp226-k16-kappa-repro-inference` version 1
- ref / Public LB:
  `54491603 / 9.837`
- submission SHA:
  `b71e15f7dc7e66f7be70db4a81d9ec72e1001ff2ba13907c3aba24938e906047`

exp434のexposed current-test `exp226_k16`をID順に比較し、最大絶対差
`<=0.001 ft`、同じexp226 source SHA、同じK16 configである場合だけ再利用する。

### LikPF

- source experiment:
  `exp069_pixiux_pf_beam_direct_submit_audit`
- kernel:
  `kentookumura/exp069-pixiux-pf-beam-direct-infer` version 3
- ref / Public LB:
  `53706005 / 9.721`
- submission SHA:
  `57d5c55c5caa1d07b6691a054116b434d63dd9f8e03c73dfb6ef45753aa8fa01`

exp434とexp069 v3でstable per-well seed、500 particles、128 seeds、
generator source/config、exposed current-test値の最大絶対差`<=0.001 ft`を
満たした場合だけ再利用する。pre-patch exp069 v2の9.877は使わない。

### 固定3-way

- source experiment:
  `exp263_last_anchor_better_candidate_confidence_pair_cache`
- kernel:
  `kentookumura/exp263-last-anchor-pair-cache-inference` version 2
- ref / Public LB:
  `54761954 / 7.800`
- submission SHA:
  `6316695197ee67c9a2aaa23754e6f2a5cf30dd0ec4ef1a018921f9ea640a1dbc`

exp263 version 3でもformula parity最大0、prediction/submission SHAがversion 2と
一致しているため、同じfixed formulaを再提出しない。

K16またはLikPFの同一性gateが不合格なら既存LBはcensusへ入れず、batch 2完了後の
次の提出可能日に同じexp434 generatorから追加提出する。係数や候補は変えない。

## 実装方針（2026-07-29承認・実装済み）

新規Jupytext percent形式
`exp434_physics_candidate_public_lb_audit_compact_selfcontained_inference.py`
を作成した。正規inference Notebookは採用承認後にだけ生成する。

exp263 inference version 3を構成参照元とし、次をNotebook上で追えるようにする。

1. source/kernel/config/SHA preflight
2. raw hidden-test inputとsample submissionのidentity確認
3. exp073 PF/Beam/LikPF、exp209 exact HMM、exp223 self-GR HMM、
   exp226 K16のhidden-safe再生成
4. 6 primitiveのfinite / row / well / ID監査
5. 5 pairと固定3-wayのfloat32 formula parity
6. 凍結manifestから1候補だけ選択
7. submission生成、prediction/submission SHA、stats、fallback rows保存

同じ実験ディレクトリのhelper importへ依存する薄いNotebookにはせず、
上位orchestrationと候補選択をセルに展開する。重い親generatorは固定Kaggle
input sourceを読む。train Notebookはno-trainingの説明用placeholderのままとする。

## Kaggle version運用

- canonical kernel:
  `kentookumura/exp434-physics-candidate-lb-audit-infer`
- GPU / internet:
  off / off
- 1 kernel versionにつき1 `selected_candidate`
- 各versionで候補ID、formula、parent source SHA、package notebook SHA、
  Kaggle version/id、prediction SHA、submission SHAをmanifestへ記録する
- technical rerunでversion番号がずれても、candidate-version manifestを正とし、
  version番号を候補IDへ暗黙対応させない
- packageは候補ごとに正のconfigから再生成し、packaged configとbootstrap内configの
  `selected_candidate`一致をpush前に確認する

## 提出batch

### Batch 1: 5 pair

1. `exp226_k16__selfgr_hmm_a070`
2. `exp226_k16__exact_hmm`
3. `exp226_k16__likpf_mean`
4. `selfgr_hmm_a070__likpf_mean`
5. `likpf_mean__exact_hmm`

### Batch 2: 未提出4 primitive

1. `selfgr_hmm_a070`
2. `exact_hmm`
3. `pf_ancc`
4. `beam_mean`

Batch 1とBatch 2は実装前に同時freezeする。Batch 1のPublic LBを見てもBatch 2を
中止、差替え、並べ替えしない。既存同一性gate不合格候補はbatch 2の後へ追加し、
1日5件を超えない。

各候補はpackage run完了、output取得、`kaggle-submit-check` PASS後に停止し、
competition submitは候補ごとの別承認を必要とする。

## 集計

候補ごとに次を記録する。

- OOF RMSE
- Public LB
- `Public LB - OOF RMSE`
- OOF rank / Public LB rank / rank差
- primitive / pair / fixed
- submission ref / status / date
- kernel version / source SHA / prediction SHA / submission SHA

全12候補完成後に、Spearman順位相関とkind別の中央値をdiagnosticとして計算する。
12候補は親を共有しpairも相関しているため、相関のp値や一般化性能を強く解釈しない。
Public LB表示が同値ならtieとして扱う。Private LBは見えるまで空欄にする。

## 成功・停止条件

- technical FAIL:
  submissionせず、候補式を変えないtechnical fixだけを別承認後に行う
- hidden rerun ERROR:
  source/timeout/pathを調査するが、candidate/weight/seedを変更しない
- candidate同一性gate FAIL:
  既存LBを流用せず同じexp434候補を追加提出する
- 12候補のLB完成:
  census完了。結果を記録してbranchを閉じる

LB順位がOOFと逆転しても、weight grid、LB上位pairの再blend、candidate subset、
selector、hard gate、postprocess、新規candidateはこの実験では禁止する。

## 実行量

- train scientific variants / model configs / trained folds / boosters:
  `0 / 0 / 0 / 0`
- parent/control再学習:
  0
- new inference candidate versions:
  9
- new competition submissions:
  通常9、同一性gate不合格時最大11
- daily schedule:
  通常`5 + 4`、最大`5 + 5 + 1`
- primitive generators per hidden run:
  exp263 Stage 1と同じ6本
- GPU:
  0

## 再現性設計

- seed:
  42 + exp073 stable SHA256 per-well seed
- stochastic:
  PF/Beam/likelihood-PF raw-test regeneration
- parallel RNG:
  well IDから独立seedを作りworker schedulingへ依存させない
- deterministic components:
  exact HMM、self-GR HMM、K16、pair/fixed arithmetic
- source freeze:
  exp263 version 3の4 generator source SHA、Stage 0 manifest SHA、
  formula-parity SHAを照合する
- SHA:
  input/source/schema、primitive content、selected prediction、
  submission、candidate-version manifestを記録する
- gzip:
  raw gzip SHAではなくdecompressed/logical content SHAを主証拠にする
- model SHA:
  非該当、trainingなし
- bootstrap:
  metadata、CPU/internet、kernel sources、embedded config、
  selected candidateをpush前に照合する
- deterministic anchor:
  candidateごとにkernel version、source SHA、prediction SHA、
  submission SHAが揃った後だけtrue

## リスク

- 提出枠:
  通常9件で2日分を使う。実装・提出承認時に当日の残り枠を再確認する。
- CV/LB:
  OOF773 wellsとPublic LB splitの分布差、Public表示の丸め、候補間相関がある。
- hidden runtime:
  exposed 3 wellsではexp263 v3が354.341秒。hidden scaleは大きいが、
  fixed blendのcode submission完走実績がある。同じgenerator scopeを増やさない。
- 再現性:
  PF seed/source/bootstrap差でcandidate identityが変わり得るため、
  source SHAとstable seedをfail-fastする。
- 重複提出:
  K16、LikPF、fixedは同一性gateを通れば再提出しない。
- LB overfit:
  12候補、式、batchを結果前にfreezeし、結果後の派生を禁止する。

## 今回の実装境界

設計セッションではsteeringとdesign-only scaffoldだけを作った。後続の実装承認で
Jupytext source、compact候補Notebook、候補生成、同一性gate、manifest、
専用testまで実装した。正規Notebook実装、Kaggle package / push / run、
output取得、submit-check、competition submissionは行っていない。
