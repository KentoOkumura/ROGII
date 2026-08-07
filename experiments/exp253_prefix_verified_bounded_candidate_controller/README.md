# exp253 prefix-verified bounded candidate controller

## 状態

- ルート: ensemble
- 状態: Stage 1完了・性能guard不通過・不採用
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-15
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- candidate親: `exp072_exp063_full_replay_feature_cache`

## 仮説

既知prefixを50%・65%・75%で短縮したlegal pseudo-holdout上で既存candidateを評価し、
gain・rank margin・cut consistencyが十分なwellだけ、exp238 baseから選択candidate方向へ
最大40%、最大30 ftの補正を加えると、hard selectorのworst-well回帰を抑えながら改善できる。

## 変更点

- 公開notebook `yusuketogashi/rogii-another-approach` のprefix candidate scoreとbalanced bounded moveだけをsource-portした。
- candidateは既存exp072の`last_anchor_tvt / pf_ancc / pf_z / beam_mean / beam_med / sc_ens / hyb / likpf_mean / tvt_dense`に固定した。
- fallback/baseは保存済みexp238 `lgb_mean` OOFに固定した。
- contact、surface/poly candidate、heel、affine GR、bimodal correction、新規学習は含めない。

## 検証方針

- Stage 0: sorted 32 wells、3 cuts、CPU single process、0 boosterでmask・candidate score・bounded move contractを確認する。
- Stage 1: 全773 wellsをstable SHA256 well modulo 4で分割する。各shardは全raw wellsをimputerへ渡し、評価wellだけを分ける。
- Group: well。fold指標はSHA256 well mod 5の診断面で、model selectionには使わない。
- Leakage check: synthetic cut後の`TVT_input`と`TVT`をcandidate generatorからmaskし、truthはprefix scoreとofficial-tail評価にだけ使う。
- Adoption guard: overall改善、near/1000+/hidden-like非悪化、3/5 folds改善。worst-well回帰はユーザー判断によりmonitor-only。

## 実行入口

- 学習/監査 notebook: `exp253_prefix_verified_bounded_candidate_controller_train.ipynb`
- Stage 1 shard notebook: `exp253_prefix_verified_bounded_candidate_controller_train_variant0.ipynb` ～ `train_variant3.ipynb`
- Stage 1 aggregate notebook: `exp253_prefix_verified_bounded_candidate_controller_train_aggregate.ipynb`
- 推論 notebook: `exp253_prefix_verified_bounded_candidate_controller_inference.ipynb`
- train package準備: `make prepare-kaggle-notebooks EXP=exp253_prefix_verified_bounded_candidate_controller EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp253-prefix-bounded-controller-train --title 'exp253 prefix bounded controller train' --run-on-push"`
- inference package準備: `make prepare-kaggle-notebooks EXP=exp253_prefix_verified_bounded_candidate_controller EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp253-prefix-bounded-controller-inference --title 'exp253 prefix bounded controller inference' --run-on-push"`
- notebook実行はKaggleを正とし、ローカル実行は行わない。

## 結果

Stage 1は4 CPU shardで773 wells / 2,319 requestsをerror 0で完走し、aggregate technical checksは9/9通過した。
全3,783,989 rowsのoverall RMSEは7.936701から8.205455へ悪化した。1000+は+0.307983、
hidden-like 2面は+0.282873 / +0.267543、foldは0/5改善だった。worst well `fcfcc902`の
+10.310641はmonitor-onlyだが、残りの必須guardでも不通過のためadoption/inferenceを不許可とする。

## 所見

- prefix candidate evaluationとbounded correctionを単一変更として実装し、全wellまで監査した。
- Stage 0の32-well改善は全wellへ転移しなかった。near 250 ftまでは小改善したが、500 ft以遠で悪化しoverallを壊した。

## 次のアクション

branchを不採用として終了する。parameter grid、inference、submissionは行わない。

## 注意

- `likpf_mean`をdefault candidateとするのは、新規selector candidateを作らずraw test再生成できる既存pathに限定するための固定仮定。
- PF/likelihood-PFはstochastic要素を持つため、Stage 0/1とも`n_jobs=1`とし、rerun SHA一致前はdeterministic anchorと呼ばない。
- inference notebookはStage 1 summaryの`adoption_supported=true`がない限り明示停止する。
- shard notebook単独は常に`inference_allowed=false`とし、4 shard aggregateだけが採用可否を決める。
- canonical短縮slug/titleと`run_on_push=true`は上記prepare CLI引数で固定する。引数なしの再prepareは行わない。
