# exp386_cycle_consistent_rgt_scenario_bank

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU version 1完了・Stage 0 FAIL_CLOSE
- CV: 未実行
- Public LB: 未提出
- 作成日: 2026-07-24
- 親実験: なし（topology-first RGT の独立系統）
- 比較対象: exp226 / exp293 / exp301 / exp377 / exp383
- 後続候補: exp387は親gate不成立により未実装で閉鎖

## 仮説

地層面を各井戸の絶対 TVT 面として局所補間するのではなく、6 地層の順序と区間内位置を相対地質時間（RGT）に変換し、outer-train 井戸間の対応をサイクル整合なグラフとして解けば、対象井戸の物理的に異なる複数の TVT 経路を生成できる。

exp377 で確認した「固定 K16・近傍50・bandwidth 500 ft の局所補間では安全に精度を検証できない」という問題を、単一面の精度改善ではなく、順序・対応関係・複数解の保持によって回避する。

## 設計

- outer-train の TVT と 6 地層面から RGT ノード・井戸間エッジ・基本サイクルを構築する。
- outer-valid/test の生の地層列、未知 suffix TVT、GR は scenario bank 固定前に読まない。
- 対象井戸では軌跡座標と既知 `TVT_input` prefix だけを使う。
- 決定論的 k-shortest path により、井戸ごとに 8〜32 本の TVT scenario を保存する。
- scenario の事後最良選択は行わず、本実験の出力は candidate bank と prior cost に限定する。
- exp387 用の参照 GR template は outer-train のみから生成する。

## 検証方針

- Fold: exp226 と同じ outer 5-fold
- Group: `well_id`
- 評価行: outer-valid の未知 suffix
- Stage 0: target-free leakage、RGT/graph/bank coverage、cycle residual、16井戸 resource audit
- Stage 1: rolling-origin prefix で exp226 に対する oracle 改善が 0.50 ft 以上、5 fold 中4 fold以上で改善
- Stage 2: truth-late scenario oracle RMSE 5.50 ft 以下、全5 foldで改善
- Stage 2 を通過しない限り exp387 に進まない。

## 計算量

- scientific variant: 1
- graph solve: 5 fold
- target-well path solve: 773
- LightGBM / HMM / PF / Beam: 0
- 親 control の再学習: なし

## 実行入口

別名の
`exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_train.py` /
`.ipynb`にtrain-side Stage 0〜2を実装した。inferenceはfail-closed候補だけを追加した。

compact候補を正規`exp386_cycle_consistent_rgt_scenario_bank_train.ipynb` /
`exp386_cycle_consistent_rgt_scenario_bank_inference.ipynb`へ採用した。
Kaggle private CPU / internet offの16-well Stage 0 preflightと、
PASS後だけのfull runを承認済み。inferenceとsubmissionは無効のまま維持する。

## 結果

canonical train kernel version 1（id_no `128478384`）は`COMPLETE`。
16-well / 5-fold Stage 0は、RGT source coverage `0.989847`、leakage 0、
projected runtime `2867.246 sec`、peak RSS `1.145931 GB`をPASSした。

一方、graph query coverage、scenario-bank well coverage、scenario count p05、
finite-path coverageはすべて`0.0`で、cycle residual p95も
`2.363303 > 0.10`だった。Stage 0はFAIL_CLOSEとし、full run、Stage 1/2、
inference、submissionは実行していない。

## 所見

### リスク / 注意

- 外挿先に十分な RGT 対応がなければ、scenario 数だけ増えても oracle は改善しない。
- 地層区間の反転や欠損は並べ替えて補正せず、利用不可として扱う。
- cycle residual、edge 数、scenario 数などの事後 rescue grid は禁止する。
- 初回成功 run は deterministic anchor とせず、同一 content SHA の再実行を要求する。

### 次

この設定のfull runとexp387は行わない。再訪する場合もedge/cycle/path設定を救済せず、
同じ固定設定でroute rejection段階とedge residual成分だけを測る0-prediction readoutを
別実験として事前設計する。
