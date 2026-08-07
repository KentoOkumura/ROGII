# 設計

## アプローチ

exp374のcompact self-contained train/inference実装をsource SHAで固定し、
current/hidden testのraw horizontal/typewell dataへ同じdeterministic exact HMMを
適用するinference Notebookを、後続の実装承認後に1本だけ作る。
候補はStudent-t emissionのposterior meanを直接提出し、Gaussian controlは生成しない。

この実験はexp374のtail FAILを再分類するものではなく、Public LB上で平均改善と
少数well tailのどちらが強く現れるかを記述するcensusである。

## 実験範囲

- 対象実験: `exp454_student_t_exact_hmm_direct_public_lb_audit`
- Route: `pf_beam`
- 親実験: `exp374_exp209_student_t_exact_hmm_emission`
- scientific control: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Public LB control: exp434 `exact_hmm`
- sibling reference: `exp389_exp209_huber_exact_hmm_emission`
- 変更する変数: なし。exp374候補をcurrent-testへ同一式で移植するだけ。
- 固定する変数:
  - absolute-TVT coordinate、grid step 0.35 ft、band pad 100 ft
  - 41 rate states、rate span contract、`sig_r=0.002`、`sig_p=0.02`
  - `lam=1.0`、`start_sig=0.75`、`r0_sig=0.01`、`momentum=0.998`
  - known-prefix zero-fill population std、sigma clip `[10,60]`
  - horizontal GR両方向線形補完、Type Well GR ffill/bfill
  - posterior mean
- 将来の実行Notebook:
  `exp454_student_t_exact_hmm_direct_public_lb_audit_inference.ipynb`の1本
- template由来train Notebookは設計scaffoldであり、実行・実装対象にしない。
- train/model/booster/PF/Beam/control rerun: `0 / 0 / 0 / 0 / 0 / 0`
- competition submission順: 3候補中3番目。LBを見て順序や式を変えない。

## 固定候補

標準化GR残差を`z`、`df=4.0`として、各TVT stateのrow log emissionを

```text
-0.5 * (df + 1) * log1p(z^2 / df)
```

とする。state-independent normalization定数は省略し、追加clip、temperature、
Gaussian mixtureは使わない。これ以外はexp209 exact HMMと同一で、最終出力は
full-sequence posterior mean TVTである。

## LB評価契約

- OOF evidence:
  `11.938287235 -> 11.720478702`、gain `0.217808533 ft`、4/5 folds。
- fixed scopes:
  raw observed `+0.131927`、raw missing `+0.404486`、high-missing `+0.503873`、
  1000+ `+0.236105`、hidden-like spatial `+0.488873`、
  hidden-like typewell-purged `+0.415856 ft`。
- tail evidence:
  by-well p95 `+0.982661344 ft`、worst `+35.015963236 ft`。
- Public LB control:
  exp434の同一exp209 semantics `exact_hmm` direct LBを使う。
- `candidate LB < exact_hmm LB`なら公開splitではStudent-t emissionを支持、
  それ以外は不支持と記述する。
- Kaggle表示3桁の同値はtieとし、差を過剰解釈しない。
- 結果にかかわらずtrain-side tail FAILは履歴として維持し、自動採用しない。

## 再現性設計

- seed policy: no RNG。
- stochastic処理の有無: なし。
- PF/Beam / likelihood-PF / seed baggingの有無: なし。exact HMMのみ。
- 並列処理:
  sorted well、sorted row、fixed grid/rate順。Numba thread countを固定する。
- CPU/GPU runtime:
  CPU-only、internet off、上限30,600秒。train 773 wells実測
  `19,662.082424 sec`をhidden runtime上限の保守根拠にする。
- input/test regeneration SHA:
  raw well manifest、schema、candidate decompressed content SHAを保存する。
- model manifest:
  modelなし。model/booster count 0を記録する。
- prediction / submission SHA:
  candidate logical prediction SHA、CSV SHA、submission SHAを保存する。
- Kaggle bootstrap:
  prepare後にbootstrap内configと正のconfigのcandidate ID、df、HMM全固定値、
  runtime、submission禁止フラグが一致することを確認する。

## 技術gate

- exp374 pinned source/config SHAが一致する。
- sample submissionから行数、ID集合、well集合を動的に解決する。
- test wellごとのprefix supportとtypewell参照を検査し、silent fallbackを禁止する。
- posterior normalization、finite coverage 100%、duplicate ID 0、fallback 0。
- Student-t branch activationとdf値をmanifestへ記録する。
- HMM well-runsはtest well数と一致し、Gaussian/Huber/PF/Beam runは0。
- submit-check PASS前および別承認前にcompetition submissionしない。

## リスク

- リークリスク:
  suffix TVT/error/fold/roleをdecode入力から排除する。Type Well TVTは参照座標として許可する。
- CV/LB不一致リスク:
  overall OOF gainは大きいが、1 foldとwell tailが悪化している。
- ランタイム/メモリリスク:
  exact HMMはwell長とgrid幅に依存する。上限見積もりを超える場合はpushしない。
- 再現性リスク:
  no RNGだがdtype、Numba、well/row order、interpolation差でpredictionが変わり得る。
- ガバナンスリスク:
  exp434 exact HMM LBを見たdf/scale調整はblind LB tuningになるため禁止する。
