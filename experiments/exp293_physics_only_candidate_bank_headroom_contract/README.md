# exp293_physics_only_candidate_bank_headroom_contract

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU audit完了・support PASS
- CV: H512 oracle RMSE `3.683763`（diagnostic upper bound）
- Public / Private LB: 対象外
- Submit ID: 対象外
- 作成日: 2026-07-19
- 親実験: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 設計の正: `.steering/20260719-exp293-physics-only-candidate-bank-headroom-contract/`
- 後続分岐の正: `downstream_branch_contract.md`

## 仮説

exp263 Stage 1でcurrent-test再生成済みの6 primitive、5 fixed pair、1 fixed formulaだけでも、
H512 block oracleは物理モデル単体Public LB 6.5を支える十分なheadroomを持つ。

## 変更点

- exp263 deployable12をcandidate値・formula・順序ごと固定する。
- row、非重複H128/H256/H512、whole-well oracleを同じbankで計算する設計を追加する。
- anchor 8.2383315465から6.5へ必要なoracle SSE headroom回収率を算出する。
- support PASSならStage 2、FAILならStage 4という分岐を固定する。
- GR score、candidate選択、decoder、candidate生成、学習、推論、提出は行わない。

## 検証方針

- Fold: exp263 outer 5 foldsを再利用しfitなし。
- Group: `well`。
- Primary: H512 pooled oracle RMSE `<=5.5`かつ全fold `<6.5`。
- Secondary: row/H128/H256/whole-well、distance bucket、1000+、hidden-like、by-well。
- Leakage check: candidate bankとblock assignmentのSHA freeze後にだけtrue TVTを別loaderでjoinする。
- Technical contract: 3,783,989 rows / 773 wells / 12 candidates / finite coverage 1.0。

## 実行入口

- canonical train: `exp293_physics_only_candidate_bank_headroom_contract_train.ipynb`。
- Jupytext source/parity候補: `exp293_physics_only_candidate_bank_headroom_contract_compact_selfcontained_train.py/.ipynb`。
- inference候補: `exp293_physics_only_candidate_bank_headroom_contract_compact_selfcontained_inference.py/.ipynb`。
- inference候補はfail-closedで、raw test・TVT prediction・submissionを作らず停止する。
- compact trainは正規trainへ採用済み。正規inferenceは未採用で、raw-test inference/submissionはdisabled。

## 実装内容

- exp263 manifest、5-fold candidate partition、logical content/schema/file SHAを検証する。
- 6 primitiveをfloat32 memmapへ読み、5 pairと固定formulaをexp263 Stage 1と同じ演算順で再構成する。
- formula sample parity、全12候補OOF RMSE parity、row identity、coverageをfail-closedで検証する。
- row/H128/H256/H512/whole-well blockをcandidate bankとともにfreezeし、その後だけraw train truthをjoinする。
- 3,783,989×12の二乗誤差wide matrixは作らず、candidate×row chunk単位でrow minimumとgroup SSEを集約する。
- pooled/fold/distance/1000+/hidden-like/by-well/choice count、固定support判定、input/bank/block/readout SHAを保存する。
- oracle/selected TVT predictionは保存しない。

## 生成物

Kaggle version 2で`config.yaml`のexpected artifacts 11件を生成した。ローカル取得物のfile SHAとgzip
decompressed SHAをmanifestに対して再計算し、不一致0。oracleまたはselected TVTのrow予測は保存していない。

## 結果

| メトリック | 値 |
| --- | --- |
| H512 pooled oracle RMSE | `3.683763` |
| H512 fold最大oracle RMSE | `4.117908` |
| 6.5への必要SSE headroom回収率 | `0.471825` |
| support判定 | `PASS` |
| Public / Private LB | 対象外 |

## 所見

### 現時点で確定した点

- primary bankはexp263 deployable12であり、exp286 `geop_hmm`などを混ぜない。
- primary粒度はH512で、末尾short blockも評価する。
- support PASS/FAILとStage 2/3/4への分岐は固定済み。
- Stage 2 FAILからStage 4へ自動分岐しない。

### 未評価の点

- Stage 2のGR evidence識別性とheadroom recovery。
- Stage 3のjoint smoother OOF。

### リスク / 注意

- oracleは到達可能性の上限であり、実際のtarget-free選択性能ではない。
- exp263 OOF core12とStage 1 deployable12を混同しない。
- Public testは3 wellsなので、exp263のCV-LB差を6.5判定の根拠にしない。

## 次

固定分岐どおりStage 2 `prefix_calibrated_latent_registration_gr_evidence`の設計へ進む。
