# Blind benchmark round 1 judgement

## 判定方法

- 候補出力の source-hidden packet 以外は発想の採点根拠にせず、judge だけが指定された一次 archive を読んだ。
- 主採点は各出力の top 5、補助採点は全12案。各 gold mechanism を 0/1/2 点（0: 該当なし、1: 部分一致または決定的要素がない、2: inference input・target/output・mechanism role・反証可能な test が揃う具体案）で採点した。
- モデル名や `U-Net`、`PF`、`heatmap` 等の単語一致には点を与えていない。別カードの断片を恣意的につないで1つの完成案とすることもしない。ただし G4 は portfolio 自体の候補源と融合を評価する。
- Gold の機構根拠は、733220/733150/733174/733305 の whole-well cost/probability field、733220 の PF heatmap input、733305/733226/733150 の self/heel reference と coverage fallback、733226 の reference・physics・dynamics が異なる候補の soft bagging、733154/733150/733315 の uncertainty gate、733220/733150/733174 の invariant-preserving augmentation、733154 の fabricated first-pass mistake refiner、733154/733226 の hidden-well runtime engineering と GPU PF/smoothing である。

## Gold mechanism 採点

| Gold | Control top 5 | Treatment top 5 | Control 全12 | Treatment 全12 | 判定根拠 |
|---|---:|---:|---:|---:|---|
| G1 whole-group probability/cost-volume + coherent decode | 2 | 2 | 2 | 2 | Control I03 と treatment I01 は、候補を row point に潰さず whole-well lattice の node、unary、transition として連続性制約付きで decode し、constrained oracle と target-free decode を分離する。dense cost volume そのものではないが、gold と同じ「全体表現を先に保ち coherent に落とす」実験可能な機構である。 |
| G2 weak heatmap/posterior を input/evidence 化 | 2 | 2 | 2 | 2 | 両者の I02 は heatmap peak/path を提出値にせず、既存 path 上の logit mass、entropy、margin、coverage を compatibility evidence に変換し、shuffled-GR control と held-out ranking/regret で検証する。gold の PF heatmap channel や selfGR-as-feature と機構レベルで一致する。 |
| G3 same-entity revealed context + coverage fallback | 1 | 0 | 2 | 2 | Control top の I01 は revealed prefix TVT residual を same-well observation にするが、self GR-vs-TVT reference と covered TVT range 外の typewell fallbackまではない。Treatment top 5 には具体的な same-well 観測案がない。全12では control I08 の short-prefix fallback、treatment I04 の known-prefix GR calibrator と invalid/short-prefix zero fallbackが具体化されるため2。ただし archive の「self-reference image/PF reference、範囲外はtypewell」がより直接的である。 |
| G4 different information/reference/dynamics candidates + soft fusion | 1 | 2 | 2 | 2 | Control top は I03 で existing paths と heatmap modesを増やす一方、同じカードの最終操作は主に hard coherent decode、I02 の soft weighting対象の候補源が別 reference/dynamics だと明示されず部分点。Treatment top は I03 が ML anchor と physical/candidate family の disagreement を非負・縮小 residual stack で soft に使い、I07 が heatmap を既存 union への complementary source として追加する。全12の control I05 は ML/HMM-PF/heatmap experts の posterior meanを明示するため満点。 |
| G5 uncertainty/disagreement-gated weak signals | 2 | 2 | 2 | 2 | Control I02/I03 は entropy・margin・coverage で weak heatmap evidence/branch を抑制し、treatment I03 は disagreement/uncertainty で correction を0へ縮小して MLへ abstainする。いずれも信号、gate、fallback、held-out test がある。 |
| G6 invariant-preserving augmentation/synthetic data | 0 | 0 | 2 | 2 | top 5 には両者とも該当案がない。全12では control I07 が prefix boundary を保つ offset/slope/regime/tail corruptionを real residual分布で調整し、treatment I09 が candidate値・ID対応を保った長区間 score inversion/availability corruptionを training foldだけに加える。後者は地質的 synthetic well ではなく selector-object augmentationだが、保存 invariant と clean held-out test が具体的なので満点。 |
| G7 realistic corrupted first-pass conditioning/pretraining/refiner | 0 | 1 | 1 | 1 | Treatment I03 は first-pass MLを残す second-pass residual correctionとしては部分一致するが、誤ったconditioningを生成して trust を学ばせる pretrainingがない。全12の control I07 / treatment I09 は realistic corruptionを作るが、学習対象は candidate scorer/selectorであり first-pass-conditioned refinerではない。I08等のcorrectorと同一実験に接続されていないため満点にはしない。 |
| G8 acceleration unlocking inference/smoothing + hidden-unit runtime | 2 | 2 | 2 | 2 | Control I11 と treatment I11 は、10h50m/14.88h の失敗を起点に O(KN) one-pass/cache/streaming、数値 parity、RSS/hash、row数・longest well・約200 wellへの外挿、7.2h目標/9h killを明記する。PF GPU portほど直接smootherを高速化しないが、従来不可能な nested scoring/hidden inferenceを解禁する具体的 enablerである。 |
| **合計** | **10/16** | **11/16** | **15/16** | **15/16** | top 5 は treatment が +1、全12では同点。 |

## 安全性・構成品質

| 項目 | Control | Treatment | 判定 |
|---|---|---|---|
| unsafe / leakage 案 | 0/12 | 0/12 | 両者とも truth-aware oracle は representation/coverage 診断に限定し、selector/calibrator/threshold は outer whole-well fold の train側だけでfitする。revealed prefixは推論時に合法な情報としてのみ使う。Treatment I07 の marginal-gain label も training fold限定。 |
| parameter-only率 | 0/12 (0%) | 0/12 (0%) | 小さな設定値を含む案はあるが、全カードに representation、information role、fusion、data distribution、validation object、compute regimeのいずれかの変更がある。Treatment I12 は incremental だが seed-paired causal identificationであり lag tuningではない。 |
| coverage/selectability 分離 | top 5: 5/5、全12: 12/12 | top 5: 5/5、全12: 12/12 | 両者とも oracle coverage/bracketing と hidden truthなしの ranking/regret/RMSEを別 test としている。Treatment は JSON schemaで `coverage_test` / `selectability_test` を全カードに強制した分、監査しやすい。 |
| top 5 mechanism family数 | 4（5つの細分類を自己申告） | 5 | Control は strict な broad familyでは information/selector、representation/candidate generation、prefix adaptation、compute の4。I02とI04は evidence scaleは違うが同じ information-selector family。Treatment は fusion/uncertainty、representation、information、candidate generation、compute の5で重複がない。 |
| source-hidden 境界 | 明示・良好 | 明示・良好 | 両者とも cutoff 2026-07-12、packetのみ、web/writeup/後続結果不使用、missing/assumption、public 3-well非依存を記録する。一次archive由来の固有手法名や事後スコアの混入は認めない。 |

unsafe と数えなかったが、実装時に再確認が必要な点はある。Control I01/I08 は既存 prediction が prefix上に定義されるという availability assumption、両出力の heatmap案は hiddenで logitsを再生成できるという assumptionを置く。ただし、いずれも最初のavailability check、停止条件、fallbackを記しており、現時点ではリーク案ではない。

## Control と treatment の差

Treatment の改善は大きな新gold発見ではなく、top 5 の構成と監査性にある。

1. Top 5 の broad family数が4から5へ増えた。Control は I02/I04 がいずれも selector evidenceで重複するが、treatment は fusion/uncertainty と complementary candidate generationを独立slotにした。
2. G4 が top 5 で1から2へ改善した。Treatment I03 は anchorと異なる物理候補familyを disagreement-aware soft residualとして使い、I07 は既存unionへの marginal coverageを狙う別情報源を組み合わせる。Control の完成した soft multi-family fusion は I05 にあるが top 5 外だった。
3. G7 は treatment top 5 で0から1へ上がった。I03 が first-pass anchorを入力にした安全な second-pass correctionを具体化したため。ただし fabricated-error pretrainingがなく、goldの核心には届かない。
4. Treatment は coverage/selectability、hidden contract、compute estimateを固定schemaで必須化し、同じ内容を持つcontrolより比較・欠落検出が容易である。

一方、treatment は control top の I01 にあった same-well revealed-prefix adaptationをtop 5から落とし、G3が1から0へ後退した。このため総改善は +1 に留まり、全12案のgold coverageは同点である。

## Skill v2 で直すべき具体的欠落

1. **G7専用カードを必須化する。** `first-pass prediction` を conditioning channel、`truth + tie-in anchored low-pass/random-walk corruption` を training input、truth trajectory/fieldをtargetとし、synthetic pretrain→短いreal OOF fine-tune→clean held-well testを1枚のカード内で接続する。selector score corruptionや通常のresidual stackingをG7満点扱いしない。
2. **same-entity contextを抽象的なprefix featureで終わらせない。** Revealed prefixから `(TVT, GR)` self-referenceを構築し、cost-volume/heatmap channelまたはPF observationとして使い、self coverage内/外を分け、外側はtypewellへfallbackする案をtop 5候補として生成させる。短prefix、重複TVT bin、missing GRのfallback testも要求する。
3. **候補多様性の出所をカードに列挙させる。** seedや近いparameterだけでなく、reference（typewell/self）、information（GR/geometry/physical prior/heatmap）、dynamics（filter/smoother/beam）のどの軸が違うか、residual correlationと既存unionへの marginal bracketing gainを記録し、soft posterior/tempered blendまで同じカードで検証する。Control topのように「existing five」の由来が曖昧な案はG4部分点に留める。
4. **top 5選定にgold coverage制約を追加する。** 今回は両者ともG6をtop 5から落とし、treatmentはG3も落とした。safe/exploration/orthogonal/computeだけでなく、同点なら未被覆mechanismを優先し、family重複を1枠までに制限する。
5. **augmentationのinvariantをドメイン量まで明示する。** Treatment I09のID/value保存だけでなく、`TVT + Z`、tie-in continuity、monotone typewell warp、GR/reference再計算、real error amplitude/correlation lengthなど、何を保存し何を意図的に壊すかを必須欄にする。
6. **compute案は enabler の下流を固定する。** One-pass/cacheが「何を新たに可能にするか」を selector full OOF、PF smoother、whole-well decodeのいずれかに結び、end-to-end accuracy runまでをaccept条件にする。両出力は概ね満たすが、smoothing本体の加速ではない点を明示するとgoldとの距離がさらに明確になる。

## 最終判定

Round 1 の勝者は **treatment（僅差）**。主指標は 11/16 対 10/16で、改善は top 5 のfamily diversityとmulti-family soft fusionにある。ただし全12案は15/16で同点であり、発想集合そのもののgold coverageが増えたわけではない。次版の最優先課題は、G7を断片ではなく「realistic corrupted first-pass conditioning → pretraining → refiner → held-well test」の閉じた実験カードとして生成すること、その次がself-reference + coverage fallbackをtop 5へ確実に残すことである。
