# 要件

## 依頼

- raw GR欠損補間の不確実性を、unknown suffix TVTやHMMを使わずknown prefixのpseudo-gapで校正する。
- 2026-07-22の実装承認に基づき、compact self-contained Stage 0 train候補、
  fail-closed inference候補、contract test、正規Notebookを実装する。
- Kaggle package/push/run、HMM、推論、提出は実装承認に含めない。2026-07-22の追加指示`実行してください`により、CPUのStage 0 package/push/runだけを追加承認済みとする。

## 仮説

exp269では欠損rowのemissionを完全に外すとRMSEが`+1.410212 ft`悪化した。一方、補間値をraw観測と同じ確度で扱うのも過信になりうる。outer-train known prefixのfinite GRへ実欠損run分布を再現したpseudo-gapを作り、補間誤差の分散をgap長`L`と最近傍raw finite距離`d`で校正すれば、補間値を保持したまま不確実性だけを推定できる。

## 制約

- Route: `pf_beam`。
- 科学的親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、negative referenceは`exp269_raw_hmm_likpf_missing_gr_observation_neutrality_ablation`、未実行参考は`exp308_imputed_gr_confidence_downweight`とする。
- 5-fold GroupKFoldのouter-train wellsだけでmissing-run histogramとuncertainty tableをfitし、outer-valid wellsのknown prefix pseudo-gapへ適用する。
- missing-run長は`1--64`へclipし、binを`1--3 / 4--7 / 8--15 / 16--31 / 32--64`に固定する。最近傍anchor距離binは`1 / 2 / 3--4 / 5--8 / 9--16 / 17--32`に固定する。
- 各well×run-length binでpseudo-gap候補をstable SHA256順に最大4件選ぶ。両側にfinite anchorがあるinterior gapだけを対象とし、同じraw rowを同一fold内で複数pseudo-gapへ使わない。
- exp209互換のlinear interpolationを使い、隠したraw GRとの差だけを評価する。
- 2D cellのMSEを`n/(n+200)`でlength-only MSEへ縮約し、length-onlyも同式でouter-train global MSEへ縮約する。
- HMM、TVT prediction、model、booster、inference、submissionは0。
- suffix TVT、error、abs_error、oracle、同一outer-valid foldのpseudo-gap errorをtable fitへ使わない。

## 受け入れ基準

- outer-valid pseudo-gap coverageが全fold90%以上、対象wellが各fold140以上、全foldを通したdistinct wellが700以上。
- 2D uncertainty tableのGaussian NLLがouter-train global constant imputation varianceよりpooledと4/5 foldsで改善する。
- predicted variance / observed MSE比がpooledで`[0.80,1.25]`、4/5 foldsで`[0.70,1.40]`に入る。
- gap長と推定`σ_imp`のSpearmanがpooled`>=0.50`かつ4/5 foldsで正。
- matched circular pseudo-gap placement controlよりreal placementのNLLがpooledと4/5 foldsで良い。
- 1条件でもFAILならexp341の実装資格を与えず、bin、support、pseudo-gap数、補間法の救済を行わない。
- RNGなし、fold、pseudo-gap identity、table、auditのdecompressed content SHAを記録する設計になっている。

## 次のアクション

canonical Kaggle CPU Notebook version 1を完了した。real-vs-circular fold gateが2/5でFAILしたため、固定規約どおり救済せずexp341を閉じる。
