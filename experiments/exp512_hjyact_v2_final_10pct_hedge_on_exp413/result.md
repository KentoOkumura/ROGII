# exp512_hjyact_v2_final_10pct_hedge_on_exp413 結果

## 状態

Kaggle inference version 8は`COMPLETE`。pre-v7 pullに保存されていたexact v6 sourceを再実行し、
canonical kernelのlatest versionとした。hjyact exact parity、exp413 numerical witness、共有DAG、
固定0.50/0.50式、current-test submit-checkをすべてPASSし、両componentとsubmissionはversion 6とbyte-identicalだった。
後にユーザーがexp512として実施したcode submission ref `55255459`は`COMPLETE`となり、Public LBは`6.541`。
exp413 / exp510の`7.201`を`0.660`改善したため、新しい全体・アンサンブルのPublic LB基準とする。
honest OOFとPrivate LBはない。Codexはcompetition submitを実行していない。

## 固定設定

- Route: `ensemble`
- 親: `exp413_scale5_likpf_full_replacement_on_exp335`
- public source: hjyact version 2 / run `337064157` / source Public LB `6.568`
- source profile: `vp_balanced_modelpkg_005`
- blend: `0.50 * exp413 + 0.50 * hjyact_v2_final`（float64）
- candidate reuse: fingerprint-identicalなdeterministic nodeだけを両consumerで共有
- route-specific PF: 別生成
- latest version 8 well loop: exact v6の逐次実装
- latest version 8 model-package correction: datasetと5 modelsを実行、p95 diff guardでeffective weight 0
- version 7速度履歴: SP45 / exp413 HMM / route PF / K16をrequested 4 jobs、model-packageを無効化
- 新規booster / 親control再学習: `0 / 0`

## 現在の結果

| 項目 | 値 |
| --- | --- |
| 候補source | 実装済み、7,236行 |
| source SHA | `16982879...1b6240f` |
| 構文 / Ruff F821 | PASS |
| Jupytext / 専用契約テスト / validate-exp | PASS / 8 passed / PASS |
| source parity | version 8 exact PASS (`b192d3f3...`) |
| exp413 component parity | numerical witness PASS (`3a9bbd1f...`) |
| version 5 K16 Haswell subprocess / runtime-fit audit | PASS / PASS (`7.06484e-9 < 1e-7`) |
| version 5 runtime | 1,187.181秒、最終予測生成後にfail-close |
| 診断CSV形式 / ID順 | PASS / exact |
| 診断CSV SHA | `b960c2b1...a713e23` |
| version 6 exp413 gate | exact SHAまたは監査済みwitness、max `0.02 ft` / RMSE `0.001 ft` |
| observed exp413 reference差 | max `0.0165 ft` / RMSE `0.000753012 ft` |
| version 6 runtime | scientific `1,571.153`秒 / total log `1,587.335`秒 |
| version 7 runtime | scientific `1,197.667`秒、v6比`-373.486秒 / -23.771%` |
| version 8 runtime | scientific `1,193.477`秒 / total log `1,205.751`秒 |
| version 8 source exactness | v6 code-cell SHA `9b5a4cae...bdabee6` / embedded candidate `66ed4f78...c18f804c` |
| version 8 exp413 runtime | `360.328秒` |
| version 8 SP45 sequential well loop / visible-prefix | `175.581秒` / `112.3986秒` |
| exp413 runtime | `297.388秒`、v6 `483.784秒`比`-38.529%` |
| exp413 route PF / K16 | `40.961秒 (-19.879%)` / `46.116秒 (-24.836%)` |
| exact / self-GR HMM | `55.449秒` / `57.535秒` |
| SP45 well loop | `303.658秒`、v6 sequential `218.912秒`比`+38.712%` |
| model-package | correction false / skipped / model 0 |
| version 8 model-package | 5 models実行、p95 `26.700659 ft > 25 ft`、guard無効化、final weight 0 |
| fixed 0.50/0.50 formula max error | `0.0 ft` |
| shared-node generation / hit | 15/15 generation=1、cache hit=1、fallback 0 |
| submission SHA | `b960c2b1...a713e23`（v6/v8 byte-identical） |
| current-test submit-check | PASS（FAIL/WARNなし） |
| visible output reproducibility | same-v6 version 6 / 8、2/2 PASS |
| intermediate byte reproducibility | WARN（deterministic-GR blockの3 well content SHA不一致） |
| hidden deterministic readiness | WARN（hidden RNG未証明） |
| submission ref / status | `55255459` / `COMPLETE` |
| submitted at | `2026-08-05 02:08:11.450000 UTC`（ユーザー実施） |
| CV / Public LB / Private LB | - / `6.541` / - |
| Public LB差 vs exp413 / exp510 | `-0.660 / -0.660`（改善） |
| Public LB差 vs source公開値6.568 | `-0.027`（改善） |
| reproducibility rerun | formal same-v6 visible component/final output 2 / 2 |

## 200 well実行時間概算

visible 3 wellの全体runtimeをそのまま提出上限と比較しない。latest version 8はexact v6 contractなので、
ユーザー指定のv6工程別200 well外挿をplanning gateの正とする。version 8の短い3 well総時間だけから下方修正しない。

| 工程 | 3 well実測・基準 | 200 well概算 |
| --- | ---: | ---: |
| SP45 128-seed PF | v6基準 `177–218秒` | `3.3–4.0時間` |
| learned test feature | `48–53秒` | `0.9–1.0時間` |
| visible-prefix / Gold | `117–143秒` | `2.2–2.7時間` |
| model package | v6基準 `43–62秒` | `0.8–1.2時間` |
| exp413生成 | v6基準 `367–485秒` | `6.8–9.0時間` |
| 合計 | — | `約14–18時間` |

この外挿では楽観下限でも9時間を約5時間超えるため、latest version 8のplanning gateは`FAIL / 要追加高速化`とする。
一方、ref `55255459`が数値score付きで`COMPLETE`したため、実際に提出されたhidden rerunがplatform実行上限内で
完走したことは実測で確認できた。Kaggle submissions APIはhidden well数と正確なinference時間を公開しないため、
`14–18時間`を実測値へ置き換えたり、逆に3 well結果から実時間を推定したりはしない。上表は保守的な
事前工程別外挿として保持する。

## 解釈

active source path、保存model/input SHA、共有DAG、dynamic ID整列、等率formula、fail-close gateは実装した。
version 3--5でsource final SHAは一致した。一方、platform依存のK16 GR posterior縮約差がexp413予測へ最大
`0.0165 ft`伝播した。version 4の係数固定、version 5のHaswell subprocess隔離でもexp413 SHAは変わらず、
CPU architectureだけではreferenceのbitwise再現に足りなかった。NumPy/OpenBLAS version差は推定候補だが、
reference環境のversion証拠がないため未確定とする。ユーザー承認のnumerical gateは未知SHAを許可せず、
v3--v5で監査済みのwitnessだけを許可したため、version 7もfail-openではない。source Public LB `6.568`と
exp512の実測Public LB `6.541`は別の記録として扱う。

version 7の速度面ではvisible 3 well全体を`23.771%`、exp413を`38.529%`短縮した。一方、
SP45のThreadingBackendはvisible 3 wellでsequentialより`38.712%`遅かった。PFのNumPy/Python処理がCPUを競合し、
thread overheadがwell並列の利得を上回った可能性が高い。したがって今回の全体短縮は、主にexp413 well並列と
model-package省略によるものと解釈し、SP45 threading自体を成功した高速化とは扱わない。latest version 8はこの
速度変更を含まないexact v6 sourceであり、200 well概算は`14–18時間`とする。3 well runtimeだけを根拠に9時間制限へ
収まるとは評価しない。提出のCOMPLETE結果は、
この保守的外挿より強いplatform-levelの完走証拠だが、工程別hidden時間の内訳を与えるものではない。

再現性面ではversion 6/8のhjyact、exp413、final submissionがbyte-identicalで、visible output gateは2/2 PASSした。
一方、candidate reuse manifestのdeterministic-GR blockは3 wellともcontent SHAが異なった。したがって最終出力の
再現性は確認済みだが、全中間生成物のbitwise一致やhidden-well RNG決定性まで拡張して解釈しない。

Public LB `6.541`はexp413 / exp510の`7.201`から`0.660`、source公開値`6.568`から`0.027`改善した。
両component scoreが同じPublic rowsでexactに再現されたという条件のもとでは、0.50/0.50三角上限`6.8845`を
`0.3435`下回っており、component誤差が完全な同方向ではなくblendで相殺されたことを示す。ただしhonest OOFと
Private LBがないので、private一般化や安定した相関構造の証拠とは扱わない。

## 次

exp512はPublic LB `6.541`の新しい全体・アンサンブル基準として記録し、canonical kernel latestはexact v6 sourceの
version 8とする。追加高速化を行う場合は、SP45をprocess分離
するか、well内128-seed bankをbatch/Numba `nogil`化する独立変更を事前検証する。今回のscored packageを自動rerun
または再提出しない。visible output 2-runは完了したが、hidden RNG監査と正規Notebook採用は別判断のままとする。static visible prediction
sidecarは採用しない。
