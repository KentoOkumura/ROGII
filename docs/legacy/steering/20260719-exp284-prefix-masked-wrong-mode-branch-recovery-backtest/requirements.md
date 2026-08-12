# 要件

## 依頼

known `TVT_input` prefixの末尾640行をpseudo suffixとして隠し、visible cutでGR尤度が支持する
誤mode branchを意図的に注入する。正しいbase branchを捨てず、exp283固定self-GR top-3と
post-event typewell / geometry evidenceだけでmodeを回復できるかをcontrolled backtestする。

今回はbacklog、実験ディレクトリ、steering、固定設計だけを作る。実装と実行は行わない。

## 2026-07-19 追加依頼

ユーザーの追加依頼により、exp283の生成物へruntime依存しないcompact self-contained実装と静的検証を
先行してよい。exp283 PASSは実装の技術的先行条件ではなく、Kaggle実行とscientific promotionのgateとして
維持する。正規stub notebookは上書きせず、別名compact `.py` / `.ipynb`を実装対象とする。

## 仮説

Assumption: 本当に必要なのは誤modeを一発で検知するoracle gateではなく、baseを常に残した
multiple-hypothesis状態で、後続区間の累積evidenceが誤branchを棄却できることである。known prefixの
未来をmaskしたbacktestなら、test-timeで利用可能な情報契約のまま回復率とfalse switchを測れる。

## scientific実行依存のoverride

- 元設計では`exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout`の全guard PASSを
  実行先行条件としていた。
- exp284はexp283生成物へのruntime / artifact依存がなく、fixed contractをself-containedに実装済み。
- 2026-07-19、依存が実験順序だけであることを説明した後のユーザー明示指示により、元のgateをoverrideし、
  exp284をstandalone controlled backtestとして1回実行する。
- version 1が評価前のtechnical input-schema mismatchで停止したため、固定contractを一切変えないversion 2を
  同じ1回のbacktestのtechnical retryとして実行する。
- この結果をexp283 PASSの代用にはしない。FAIL後のparameter rescueとdecoder接続禁止は維持する。

## 制約

- 実験名: `exp284_prefix_masked_wrong_mode_branch_recovery_backtest`
- Route: `pf_beam`
- maskは各wellの最後のcontiguous finite `TVT_input` rowから640行を固定で隠す。
- cut以前にvisible rowが512行以上あり、640 masked rowsが連続して存在するwellだけを対象とする。
- mask後128行をwrong-mode持続 / proposal観測区間、次の256行をprimary verifier、512行を
  diagnostic verifierとして使う。primary horizonは256行に固定する。
- cutより後の`TVT_input`とraw true `TVT`は、branch/evidence freeze前にgeneratorへ渡さない。
- cut直前128 visible rowsでexp280の固定shift bankをscoreし、`|shift| >= 10 ft`のlocal modeのうち
  尤度最大をwrong modeとして注入する。local maximumがなければ同条件の尤度最大を使う。
- safe baseはlast visible TVTをexp226 geometry incrementで延長し、常にbranch bankへ保持する。
- wrong active、safe base、exp283契約のself-GR top-3以外のbranchを追加しない。
- future selectorはexp283と同じtypewell evidence / geometry vetoを使い、weightやthresholdを変えない。
- no-injection paired controlとstable shuffled self-GR controlを必ず作る。
- 0 booster。HMM/PF再生成、モデル学習、補正prediction、raw-test inference、submissionは対象外。

## 固定backtest契約

- `cut = last_known_row - 640`とし、`cut+1 ... last_known_row`を全readerでmaskedにする。
- wrong branchは`safe_base + selected_shift`のconstant-offset pathとする。
- event rowは`cut + 128`。self-GR proposalはeventまでのGRだけを使い、future evidenceは
  `event+1`以降だけを使う。
- branch policiesを固定比較する。
  1. `wrong_active_only`
  2. `safe_base_plus_wrong`
  3. `safe_base_plus_wrong_plus_selfgr_top3`
  4. `safe_base_plus_wrong_plus_shuffled_selfgr_top3`
  5. `no_injection_base_plus_selfgr_top3`
- checkpointは128 / 256 / 512行、primary decisionは256行とする。earliest recoveryは
  checkpointでbaseまたはtruth-best branchが選ばれ、その後のcheckpointでも戻らない場合とする。

## 受け入れ基準

- technical guard:
  - eligible wellsが100以上、5 foldsすべてに存在する。
  - masked suffix、branch、future evidenceのfinite / identity coverageが1.0。
  - post-cut truth access before freezeが0。
  - injected shiftは全件`abs >= 10 ft`で固定bank内。
- pairwise verifier guard:
  - safe-base vs wrong-activeのH256 score-margin AUCが5/5 foldsで`>=0.60`。
  - H256 pairwise safe-base choice accuracyがpooled`>=0.60`かつ5/5 foldsで`>0.50`。
- recovery guard:
  - full branch setのH256 selected RMSEがwrong-active-onlyよりpooled`>=0.10 ft`改善し5/5 foldsで改善。
  - full branch setが`safe_base_plus_wrong`よりH256 RMSEを`>=0.02 ft`改善し、3/5 folds以上で改善。
  - H512でH256 selected RMSE gainを失わない。
- safety guard:
  - no-injection controlで、post-freeze base unique-best時のfalse switch率が`<=0.05`。
  - full branch setがshuffled-self-GR controlよりH256 RMSEで良く、5/5 foldsで悪化しない。
- fold、hidden-like、injection shift、proposal source/orientation、H128/H256/H512、recovery time、
  persistent recovery、false switch、by-well/worstを記録する。
- 全guard PASSだけが3番目backlogのdecoder実装検討を許可する。FAIL後のcut/mask/shift/K/horizon/
  likelihood/veto gridは禁止する。

## 今回の完了条件

- steering、実験ディレクトリ、config/docs/metricsに上記契約が未記入項目なしで固定されている。
- 元のexp283 gateと明示override、640-row mask、128-row observation、K=3、H=256、paired controls、
  guardが明記される。
- notebookはtemplate stubのまま、実装、Kaggle package、実行を行わない。
- backlogとexperiment summaryへ設計済み未実装として登録する。
