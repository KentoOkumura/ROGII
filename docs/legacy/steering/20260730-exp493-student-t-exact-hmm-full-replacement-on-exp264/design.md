# 設計

## アプローチ

exp264のcandidate IDを増やさず、`exact_hmm`というsemantic slotの値だけを
Gaussian exact HMMからexp374 Student-t exact HMMへ置換する。
`exact_hmm`を親に持つ2本の50:50 pairと固定3-wayも新しい値から再計算する。
残り8候補は保存exp263値との完全parityを要求する。

これはexp388の「Student-tを13本目に追加」と異なる。exp388ではStudent-tが18.3%の行で
top1になった一方、既存候補を含む順位変化とtail悪化が生じた。今回はcandidate count、
one-hot幅、ID順、legal domainをexp264と一致させ、候補集合拡張という交絡を除く。

## 実験範囲

- 対象実験: `exp493_student_t_exact_hmm_full_replacement_on_exp264`
- Route: `ensemble`
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- 物理候補親: `exp374_exp209_student_t_exact_hmm_emission`
- negative reference: `exp388_exp374_fixed13_dual_selector_on_exp264`
- 変更する変数: `exact_hmm` semantic value sourceとそのnative confidence
- changed 4:
  `exact_hmm`、`exp226_k16__exact_hmm`、`likpf_mean__exact_hmm`、
  `exp226_w500_50_50`
- unchanged 8:
  K16、self-GR HMM、LikPF、ANCC PF、Beam、K16×self-GR、K16×LikPF、
  self-GR×LikPF
- 固定する変数: candidate ID/order/domain、outer/inner folds、sample keys、
  88列feature schema、2 objectives、LightGBM parameters、gate

primary hard readoutはparentと同じ11候補domainで比較する。
fixed fallback 7候補domainも`exact_hmm`と固定3-wayの値が変わるため、
結果は原因分解用のreport-onlyとし、parent parityは要求しない。

## 評価

保存exp264 corrected Stage C v6 hard OOF `8.652531955610227`をcontrolにし、
再学習しない。2026-07-31に承認されたcompact候補ではStage A再構築と
strict nested Stage Cだけを実装する。

- 1 variant
- `pred_abs_error` / `p_within10`の2 objectives
- outer 5 x inner 4
- 合計40 CPU selector booster
- 25 compact partitions / 18,919,945 compact rows
- 45,407,868 outer-valid candidate-score rows

technical/leakage/score guardを全PASSしたうえで、scientific gateは次をANDで判定する。

- hard primary pooled RMSEがparent以下
- parentより改善するfoldが4/5以上
- near 0--250、1000+、hidden-like 2面の悪化が各+0.02 ft以内
- by-well delta p95とworst-well悪化が各+0.25 ft以内

Student-t依存候補のtop1率はreport-onlyとし、同一OOF上でusage thresholdやweightを選ばない。
PASSしてもこの実験内でdownstream、推論、提出へ進まない。

## Leakage契約

1. exp374からallowlist 6列だけを読み、decompressed SHAを検証する。
2. `well_id,row_idx`でglobal joinし、exp263のouter foldへ再分割する。
3. 候補ID/order/formula、feature schema、fold、sample keyをtruth join前に固定する。
4. Gaussian control、LikPF control、source fold、scope、gate、truth/errorはfeatureにしない。
5. outer-valid wellをinner fit、early stopping、calibrationへ入れない。
6. outer-train compactはinner OOF、outer-valid compactはinner ensembleだけで生成する。

## 再現性設計

- seed policy: selector LightGBMはseed 42。samplingはstable SHA256 keyで固定する。
- stochastic処理: LightGBM row/column samplingとcandidate-long row samplingのみ。
- PF/Beam/HMM再生成: なし。SHA固定済みsaved OOFを読む。
- 並列処理: sampling keyを並列fit前に確定し、worker内global RNGへ依存しない。
- runtime: Kaggle private CPU、`n_jobs=8`、GPU 0、internet off、上限14,400秒。
- SHA: exp263 manifest/catalog、exp264 schema/control score、exp374 decompressed content、
  rebuilt feature content、model manifest、outer-valid score、hard OOFを記録する。
- gzip: raw `.csv.gz` SHAだけでなくdecompressed content SHAを主証拠にする。
- deterministic anchor: 初回runだけでは指定せず、同一入力・設定の独立再実行一致後に限る。
- Kaggle bootstrap: package時にsource/config/contract SHAとkernel sourceを再検証する。

## リスク

- Student-t単体の平均改善がselector hard pathへ転移しない可能性が高い。
- exp374はby-well p95 `+0.982661 ft`、worst `+35.015963 ft`で、Huberよりtail riskが大きい。
- 4候補を同時に再計算するため、個別候補とformula効果は完全には分離できない。
  ただしsemantic replacementの整合性を優先し、formulaを旧Gaussianのまま残さない。
- exp388の悪化が候補数ではなくselector自体の不安定性なら、12本置換でも改善しない。
- 3.78M rows x nested 40 modelsのCPU時間とメモリ負荷がある。
- 同一OOFでのweight、threshold、domain、gate救済は過適合になるため禁止する。
