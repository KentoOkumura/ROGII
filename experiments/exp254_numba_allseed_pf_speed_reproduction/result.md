# exp254 結果

## 状態

Kaggle CPU probe v1を完了し、全guardが通過しましたが、実用的な高速化にならないため
**完了・不採用・branch closed**とします。773-well `full_workload`は実行しません。

## Kaggle実測

- kernel: `kentookumura/exp254-numba-allseed-pf-speed-reproduction-probe` v1（id_no `127307789`）
- workload: target-free固定3 wells、14,450 eval rows、particles 500、seed数`1/4/16/32/64/128`
- wall runtime: 436.888720秒
- peak RSS: 683.09375 MiB
- model config / fold / booster: 0 / 0 / 0
- GPU / inference / submission: なし / なし / なし

128 seedsの3 wells合計では、legacy Python seed loopが80.897349秒、Numba all-seedが
81.754755秒でした。`legacy / all-seed = 0.98951x`で、all-seed化だけでは高速化せず、
約1.06%遅い結果です。well別ratioも`0.98720–0.99052x`でした。
single-seed本体は既にNumba compiledで、Python call overheadが500 particles × 全rowのPF計算に対して
小さいため、外側loopをJITへ移しただけでは計算量を減らせなかったと解釈します。

一方、保存済み128-seed bankから300 candidateを再集約するwarm generationはwellあたり
0.025540–0.040417秒、3 wells合計0.104562秒でした。同じwellのall-seed PF coreとの
計算時間比は706.47–888.36倍です。これは「PFを一度計算した後の多数candidate再評価が軽い」
ことを支持しますが、PF core自体の300倍高速化を示す値ではありません。

## Parity・決定性

- legacy vs all-seed: trajectory、log-likelihood、final mean、ESS、resamplingが全well・全seed数でexact
- exp243保存済みfloat32 mean: 3 wellsすべてexact
- all-seed repeat SHA、cache round-trip content SHA、300-candidate repeat SHA: すべてexact
- 最大絶対差: trajectory / log-likelihood / final meanのすべてで0.0
- probe summary SHA256: `4898d7f60e6639139981654c7fc9818c1e24dd83f677d031220d56bd52d1704d`

Python側のsingle-seed Numba callをall-seed callへまとめる変更には速度上の採用根拠がありません。
cached seed bankの再集約は技術的には軽量ですが、現在の推論では固定集約を1本使えばよく、300候補を
生成する用途がありません。exp252でもseed候補のselectability gateが弱かったため、共通基盤としても
採用せず、300 candidate探索、追加all-seed最適化、後続実験を行いません。

## 773-well投影

3 wells / 14,450 rowsから3,783,989 rowsへ行数比例で外挿した値は次のとおりです。

- legacy PF core: 21,184.406828秒
- all-seed PF core: 21,408.933885秒
- 300-candidate warm generation: 27.381493秒
- all-seed + warm generation: 21,436.315378秒（約5時間57分）

これは`projection_from_three_fixed_length_quantile_wells`であり、773 wellsの実測runtimeでは
ありません。2–3分のend-to-end高速化はprobeで再現されず、約6時間のfull workloadを追加実行しても
現在の用途につながらないため、実測せずbranchを閉じます。

## 判定契約

legacy seed loopとNumba all-seedのper-seed trajectory / log-likelihood / final meanがexact、
saved exp243 float32 meanがexact、all-seed repeat・cache round-trip・warm candidate repeatのSHAが
一致した場合だけruntime比較を採用します。1つでも不一致なら、速度にかかわらず後続PF基盤へ採用せず、
full workloadも実行しません。

本runは全guardを通過したため測定結果自体は信頼できます。ただし不採用判断後に
`probe_summary_expected_sha256`を空へ戻し、`full_workload`をfail-closedにしました。

## 最終判断

- all-seed高速化: 不採用
- cached 300-candidate再集約: 現在の用途がないため不採用
- 773-well full workload: 実行しない
- 後続実験 / inference / submission: 行わない
- branch: closed

## 提出

精度・推論・提出を扱わない基盤実験のため、submissionは生成しません。
