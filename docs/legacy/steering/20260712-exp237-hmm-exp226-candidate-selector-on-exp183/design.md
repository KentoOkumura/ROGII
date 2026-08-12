# 設計

## アプローチ

exp183のwide base frame、dense enrichment、cluster/prior confidenceを維持する。exp209 exact HMM、exp223 self-GR HMM、exp226 K16 OOFを`id`/`well`で一対一結合し、追加候補値とtarget-free confidenceを作る。

候補は11本。各`(row, candidate)`をlong frameに展開し、LightGBM L1回帰で絶対誤差を推定する。OOF predicted-errorの最小候補をrow-wise selectorとし、そのlocal costを固定Viterbiへ渡す。Viterbiは候補switchとpath jumpだけを罰し、true TVT/errorを使わない。

## 実験範囲

- 対象実験: `exp237_hmm_exp226_candidate_selector_on_exp183`
- Route: `ensemble`
- 親: exp183、exp158、exp209、exp223、exp226
- 変更する変数: 候補集合を8→11、HMM/geometry confidence、rankerをerror head 1本に限定、Viterbi ruleを1本に固定。
- 固定する変数: exp183のbase/dense/cluster-prior surface、GroupKFold、likPF default、parent/control再学習なし。raw-testはユーザー承認済みのartifact-only固定Viterbiに限り、competition submitはしない。

## 再現性設計

- seed policy: GroupKFold=42、LightGBMとcandidate-long subsampleはfold由来の局所seed。
- stochastic処理: exp237の新規処理はLightGBM histogramとrow subsampleのみ。PF/HMM/K16は固定済みOOFを読む。
- PF/Beam / seed bagging: upstreamのstable OOFを使い、exp237で再生成しない。
- 並列処理: feature結合に並列RNGなし。CPU LightGBMの結果はtrain-side auditとして扱い、deterministic submission anchorとは呼ばない。
- SHA: gzip候補cacheはdecompressed content SHAを主証拠にし、source、feature schema、model manifest、OOF predictionをsummaryに保存する。
- Kaggle bootstrap: prepare後にbootstrap内configとkernel sourcesを確認してからpushする。

## リスク

- リーク: すべての候補はgroup-safe OOFであり、target/errorはrankerの学習labelと評価だけに使う。source contract検証に失敗したら停止する。
- CV/LB: raw-test artifact v2は14,151 rowsで実行したが、exp109/114 OOF-only long feature 320本をmedian / zero fallbackしたためfeature parityは未達。CVだけで提出しない。
- メモリ: 11候補はlong rowsを37.5%増やす。120k input rows/fold capと50k chunk predictionを保持する。
- source: Kaggleで複数kernel sourceを使う。prepare/push前にartifact filenameとkernel source availabilityを確認する。
