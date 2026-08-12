# 要件

## 依頼

予測開始から遠い `md_since >= 1000 ft` のlong-tail行について、同じhorizontal wellの
prediction zone内で、prediction start寄りにあるGR motifとの再一致を検出し、両位置が
同じTVTに近いというloop-closure制約として利用できるかを監査する。

この段階では実験ディレクトリと設計だけを作成し、matching実装、notebook実装、Kaggle実行、
補正、推論、提出は行わない。

## 仮説

Assumption: long-tailの大誤差には、一度生じたTVT offsetを後続行まで引きずるケースが含まれる。
prediction start寄りのunknown区間とlong-tail区間が同じGR motifを通過している場合、より早い区間を
擬似anchorとして使うことで、将来のsoft correctionに利用できる相対TVT情報を得られる。

## 制約

- 実験名: `exp282_longtail_prediction_zone_self_gr_loop_closure_readout`
- Route: `pf_beam`
- 科学的親実験: `exp090_lateral_self_gr_match_pseudotail_probe`
- 固定比較基準 / 将来donor候補: `exp263_last_anchor_better_candidate_confidence_pair_cache` の
  fixed formula `exp226_w500_50_50`（OOF RMSE 8.238331 / Public LB 7.800）。
- 並行比較参照: `exp281_exp226_residual_offset_exact_hmm_transition_probe`。exp281の結果は本readoutの
  実装・実行の先行条件にしない。soft correctionを設計する段階でのみ両結果を比較する。
- receiverは `md_since >= 1000 ft`、donorは同じwellのprediction zone内かつ
  `0 <= md_since < 500 ft` に固定する。known `TVT_input` prefixはdonorに使わない。
- edge生成にはraw horizontalの`MD`、`GR`、`TVT_input`、row index、well idだけを使う。
- raw horizontalのtrue `TVT`、target、error、oracle、exp263予測はedgeとconfidenceを凍結して
  content SHAを確定するまで読まない。
- 初回は0-booster diagnosticとし、LightGBM / CatBoost / XGBoost / HMM / PFを学習・生成しない。
- hard equality、donor TVTの直接copy、direct replacement、補正予測、submissionを生成しない。
- 実装時はJupytext percent形式のcompact self-contained train notebookを正とする。同じ実験内の
  helper importは使わず、inference notebookはfail-closedにする。
- 再現性は`docs/06_reproducibility.md`に従う。real matchingはRNGなし、shuffled negative controlだけ
  stable SHA256 local seedを使い、global RNGやthread schedulingに依存させない。

## 初回readoutの受け入れ基準

- 3,783,989 OOF rows / 773 wellsのcanonical row identityと5 foldを保持できる設計である。
- eligible receiver centerに対するtop edge生成coverageと有限score coverageが1.0である。
- edge、confidence、coverageをtrue TVT attachment前に保存し、logical content SHAを固定する。
- overall / fold / confidence bucket / hidden-like / by-wellで、real edgeとstable shuffled edgeを比較する。
- `abs(TVT_receiver - TVT_donor) <= 2/5/10 ft`、absolute delta TVT、coverage、forward/reverse別、
  segment-support別を記録する。
- primary high-confidence bucketはtarget-free confidenceのwell内top 10%に事前固定する。
- scientific guardは次に固定する。
  - high-confidence `within10` precisionがpooledで0.60以上。
  - all-edgeとhigh-confidenceの`within10` precision lift vs shuffledが各5/5 foldsで正。
  - high-confidenceのmedian absolute delta TVTがshuffledより5/5 foldsで小さい。
  - high-confidence receiver coverageがlong-tail receiverの1%以上。
  - post-freeze donor-transfer readoutがexp263 fixed receiver baselineを5/5 foldsで改善し、
    pooled RMSE gainが0.10 ft以上。
  - hidden-like spatial / typewell-purgedの両方で`within10` liftが正。
- guard PASSは別実験で弱いsoft correctionを検討する許可に限る。FAIL後のwindow、stride、
  confidence weight、threshold救済探索は行わずbranchを閉じる。

## 今回の完了条件

- `docs/legacy/steering/20260719-exp282-longtail-prediction-zone-self-gr-loop-closure-readout/` の
  `requirements.md`、`design.md`、`tasklist.md`がTODOなしで確定している。
- `experiments/exp282_longtail_prediction_zone_self_gr_loop_closure_readout/` がtemplateから作成され、
  `config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`がplanned状態を示す。
- active audit variant 1、LightGBM config 0、trained fold 0、booster 0、HMM/PF well-run 0、
  parent/control再学習0を記録する。
- notebookとmatchingコードはtemplate stubのままで、実装済みと誤認できる処理を追加しない。
- `experiment_summary.md`へplanned実験として登録し、個別実験化済みの項目を
  `KAGGLE_DIRECTION.md`の未着手バックログから外す。

