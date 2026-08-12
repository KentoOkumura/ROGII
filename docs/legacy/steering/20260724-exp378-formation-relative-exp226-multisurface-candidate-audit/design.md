# 設計

## アプローチ

exp226のK16 rate priorだけをexp377の6 formation-relative rateと固定median rateへ置換し、exp226の残りの物理処理をそのまま通す。候補順を `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, median6` に固定する。

評価は二段階とする。まずmedian6を唯一の直接primaryとしてexp226と比較する。次に7候補をすべて固定12候補bankへ追加し、oracle上限・unique-best・相関から「同じ情報の微小reweight」に留まらないかを判定する。

## 実験範囲

- 対象実験: `exp378_formation_relative_exp226_multisurface_candidate_audit`
- Route: `pf_beam`
- 親実験: `exp226_tvt_slope_kriging_hmm`
- 前提実験: `exp377_formation_relative_k16_slope_identifiability_readout`
- 変更する変数: exp226 K16 rate priorを7種類へ置換する。
- 固定する変数: fold、donor kernel、exp226 downstream、評価scope。
- 実行量: 7候補×5 fold=35決定論的候補run、0 booster。

## 段階と停止条件

1. exp377が不合格なら開始しない。
2. Stage 0で形状・順序・read guard・SHAを検証する。
3. Stage 1でmedian6の直接精度を判定する。
4. Stage 2で7候補をまとめてadd-onlyし、新規性を判定する。
5. Stage 2不合格ならexp379、exp380、exp382を停止する。

## 再現性設計

- seed policy: 乱数なし。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: well idと候補順を固定して集約する。
- CPU/GPU runtime と deterministic flags: CPUのみ。
- train cache / test feature regeneration の SHA 記録方針: exp377入力、7候補schema、fold別content SHA、候補bank manifestを保存する。
- model manifest / prediction / submission SHA 記録方針: model/submissionなし。candidate prediction SHAを保存する。
- Kaggle package bootstrap 確認方針: 実装時にoffline smokeを行う。今回はpackage化しない。

## リスク

- リークリスク: exp377 artifactのfold roleを誤流用しないようfold manifestとrole SHAを照合する。
- CV/LB不一致リスク: 7候補から都合のよい面を選ぶと過適合するためprimary選択を禁止する。
- ランタイム/メモリリスク: 7全列を一度に複製せずfold単位で保存・集計する。
- 再現性リスク: 候補列順とmedianのtie規則を固定する。
