# 設計

## アプローチ

exp497 Stage Eで得たpublic-core係数5本はすべて正で、独立trajectoryの平均的相補性を示した。
一方、pooled gainは`0.010314644 ft`、nonworse foldは`3/5`、hidden-likeとwell-tailは悪化し、
事前promotion gateはFAILした。本実験はその判定を変更せず、ユーザーが最終提出2枠の一方を
exp413単独にしないと決めたことを根拠に、exp497の事前固定deployment係数中央値だけを使う
reference-only portfolio candidateを作る。

## 実験範囲

- 対象実験: `exp509_exp413_strict_public_core_final_slot`
- Route: `ensemble`
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- auxiliary: `exp497_strict_public_core_fold_safe_ensemble_on_exp413`
- 変更する変数: 最終予測へstrict public-coreを`0.13716473330712417`だけ混合する。
- 固定する変数:
  - exp413 current-test prediction。
  - exp497 strict public-core current-test component。
  - exp497 meta-fold係数`[0.1719400963, 0.0878258709, 0.1346522859, 0.1371647333, 0.2136710223]`。
  - deploymentは5係数中央値、full-OOF再fitなし。
  - postprocess、router、well別weightなし。
- 学習量: scientific variant 1、model/config/fold/booster/PF/Beam/GPUすべて0。保存predictionだけをCPUでblendする。
- 状態ラベル: `design_complete_implementation_not_started_reference_override`。

## 入力契約

- exp413:
  - source: 保存済みcurrent-test prediction。
  - raw gzip SHA候補: `52ffb49110673f90b9b83b2e296e09b4ad0839164eda9ec13a91859937ebf136`。
  - decompressed SHA候補: `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`。
- exp497:
  - source: Stage I version 4で完成したstrict public-core saved model/prediction-only output。
  - Stage E weight SHA: `5e5dfc1f6adff2b433118adc8083cf0652e8a6b8725942b17dfa84882d91b7ba`。
  - model-set SHA: `dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626`。
  - strict prediction SHA: `27641aa6d28204a855b38e4debf0059031727b701066df75e19dad9902378885`。
  - source/model/schema/prediction SHAが欠ける場合はblockし、代替public outputへ差し替えない。
- sample submission:
  - 実行時にcompetition sourceから動的に読み、固定14,151行/3 wellsをassertしない。

## 評価と判定

- 科学的CV判定はexp497の既存FAILを正とし、再評価しない。
- 実装後に行うのはtruth-free technical auditのみ。
- 記録する差分:
  - final vs exp413のRMSE、MAE、p95/max absolute difference。
  - well別RMSE difference magnitude、score-zone horizon bucket別差分量。
  - start-row correction、prediction range/mean/std。
- これらの差分を見て係数を変更しない。
- technical contract PASS時だけ提出第1枠candidateを生成可能とする。外部提出は別承認。
- exp497 v1で観測したfloat32/runtime差はstrict `0.002 ft`、dynamic exp413と既存blend
  `0.02 ft`のcomponent別historical-equivalence契約で監査する。exp509 final式自体は
  float64 max差`1e-12`を維持する。

## 再現性設計

- seed policy: `no_rng_saved_prediction_fixed_float64_id_order`
- stochastic処理の有無: 本実験内はなし。上流exp497がstochasticであるため、その確定prediction SHAを入力境界とする。
- PF/Beam / likelihood-PF / seed baggingの有無: 本実験内0。上流で再実行しない。
- 並列処理と乱数の関係: 並列reductionなし。ID sortとsample order復元を固定する。
- CPU/GPU runtime: CPU、internet off、GPU 0。model inference 0。
- train cache / test feature regenerationのSHA: featureを再生成せず、2 prediction logical SHAとID-order SHAを記録する。
- model manifest / prediction / submission SHA: 上流model manifestを参照し、blend prediction content SHA、sidecar SHA、submission SHAを保存する。
- Kaggle package bootstrap: package生成時にroot configとbootstrap内config、source SHA、weight、input kernel source/versionが一致することをreadbackする。
- deterministic anchor: 初回runだけではfalse。同一入力SHAでのrerun prediction/submission SHA一致後だけtrue候補とする。

## リスク

- リークリスク: strict public-coreはpublic固有overlayを除外済みだが、入力取り違えでfull public outputを読む危険がある。component column名とsource SHA allowlistで遮断する。
- CV/LB不一致リスク: exp497はpooled小幅改善でもfold/hidden/tail gate FAIL。reference-only hedgeとして明記し、ML/ensemble anchorを更新しない。
- ランタイム/メモリリスク: 保存prediction 2本のstreaming joinだけで低い。
- 再現性リスク: exp497 Stage I outputの確定待ち。完成SHAがない状態で実装を進めない。
- 運用リスク: exp509結果を理由にweight再調整、exp497再学習、Gold/contact追加をしない。

## 次のアクション

候補実装の静的検証後、別承認がある場合だけ正規inference notebookへ採用し、Kaggle packageの
embedded config/source/support fileをreadbackする。Kaggle runと外部提出はさらに別承認とする。
