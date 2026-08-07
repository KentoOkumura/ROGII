# 設計

## アプローチ

exp371のfixed13 selector実装を構成参照元とし、13本目だけを
`exp333_segment_offset`から`exp355_dip_rate_hmm`へ置き換える。exp355 OOFは
`well_id,row_idx,fold,candidate_tvt`だけをallowlistで読み、canonical keyへ揃えてから
exp263 selector foldへrepartitionする。Stage Aで13候補用schemaを再freezeし、
Stage Cのnested selectorだけを40 CPU boosters学習する。

## 実験範囲

- 対象実験: `exp373_exp355_fixed13_dual_selector_on_exp264`
- Route: `ensemble`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp355_exp226_dip_rate_prior_on_exp209`
- 変更する変数: candidate inventoryへ`exp355_dip_rate_hmm`を1本追加
- 固定する変数: fixed12値・順序、fixed fallback 7本、既存formula、outer/inner fold、
  objective、sampling、LightGBM config、seed、context、parent score
- primary domain: 既存11候補 + `exp355_dip_rate_hmm`
- fixed fallback domain: 既存7候補のまま
- 学習量: 1 variant × 2 objectives × 5 outer × 4 inner = 40 CPU boosters
- control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / false / false / false`

## 再現性設計

- seed policy: seed 42。sampling seedはstage/fold/objectiveのimmutable keyから
  SHA256で生成する。
- stochastic 処理の有無: selector samplingとLightGBM。global RNGには依存させない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済みfixed12と
  deterministic exp355 OOFを入力に使う。
- 並列処理と乱数の関係: samplingは事前固定key seed、LightGBMは
  `deterministic=true`、`force_col_wise=true`、`n_jobs=8`。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU/internet off。
- train cache / test feature regeneration の SHA 記録方針: exp263 manifest/catalog SHA、
  exp355 raw SHA `28da6ffb...c960a41`、gzip decompressed content SHA
  `3c49f25e...3ede3`、upstream logical prediction SHA `634303f0...5e21`、
  parent exp264 score SHA、feature schema SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 40 selector model SHA、
  outer-valid candidate score SHA、hard selector OOF SHAを記録する。inference/submissionは
  今回対象外。
- Kaggle package bootstrap 確認方針: prepare後にembedded `config.yaml`、helper、
  kernel sources、CPU/internet/run-on-push、notebook/support ZIP SHAを照合する。

## リスク

- リークリスク: exp355 OOFに同居するtruth/parent/error列をfeature freeze前に開くと
  leakageになる。allowlist loaderとtruth-access counterで禁止する。
- foldリスク: exp355 saved-exp226 foldとexp263 selector foldは異なる可能性が高い。
  source foldはprovenance-onlyとし、global key join後のselector-fold repartitionを
  manifest化する。
- CV/LB不一致リスク: exp355はpooled 5/5 fold改善だがhidden-like 2面とwell tailが悪化。
  13番目追加でも平均改善とwell安全性が分離し得るため両方を保存し、自動推論化しない。
- ランタイム/メモリリスク: 13候補long tableは約49.2M outer-valid行。
  exp371と同じchunk/row capを維持し、想定約2時間のKaggle CPU実行とする。
- 再現性リスク: gzip raw SHAはmetadataで変わるためdecompressed content SHAを主証拠にする。

## 実装形

- 正規notebookは既存ファイルを上書きせず、compact self-contained train/inferenceを
  別名候補として実装する。
- 重いcandidate-long feature assemblyとnested LightGBM本体は既存共通
  `src/candidate_selector_pipeline.py`を使う。
- exp355固有のallowlist loader、SHA、global key join、repartition、paired readoutは
  `src/exp355_fixed13_candidate_cache.py`へ分離する。
- inferenceはStage C PASSだけでは開かず、downstream TVTの別承認・学習までfail-closedとする。
