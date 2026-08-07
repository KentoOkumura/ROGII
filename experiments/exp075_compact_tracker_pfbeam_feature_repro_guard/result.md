# exp075_compact_tracker_pfbeam_feature_repro_guard Result

## Status

PF/Beam feature generation v4 completed on Kaggle CPU with the stable-seed patch. LightGBM train v2 completed on Kaggle GPU from the v4 feature output. GPU inference v3 completed from train v2, submit-check passed, and the user submitted it twice with matching Public LB `8.489`.

Important correction: feature generation v3 itself was produced before the stable-seed patch. It is therefore not valid reproducibility evidence. Train v1 and inference v1 are also not valid reproducibility evidence. The current valid compact chain starts at PF/Beam feature generation v4, then LightGBM train v2, then GPU inference v3.

## Evaluation

Train feature generation completed:

| item | value |
| --- | --- |
| kernel | `kentookumura/exp075-compact-pfbeam-features-train` v3 |
| accelerator | CPU (`enable_gpu=false`, `machine_shape=None`) |
| rows / wells / features | 3,783,989 / 773 / 65 |
| raw gzip SHA | `280e94c8a7256e455ab5e3595096e56e4baf2c8e69083fc740aba48c818caa0a` |
| decompressed CSV content SHA | `d8bf4a133599f0c822335d66783cbffed74a244fd442585cb9b83d2d5b481e7c` |
| feature generation seconds | 9,793.583 |
| total notebook seconds | 10,450.066 |

This v3 train feature artifact is retained as history only. It used pre-patch PF/Beam generation and should not be used as a deterministic source.

Stable-seed train feature generation v4 completed:

| item | value |
| --- | --- |
| kernel | `kentookumura/exp075-compact-pfbeam-features-train` v4 |
| accelerator | CPU (`enable_gpu=false`, `machine_shape=None`) |
| rows / wells / features | 3,783,989 / 773 / 65 |
| raw gzip SHA | `7a091800aeb068bfa5fceba4d6331a61945afcef2197e7fb973b30287dca9a91` |
| decompressed CSV content SHA | `047b80b32e64b595f2a75e7593ecb513e1f27d43de87614dd2de82dae416d5b4` |
| feature generation seconds | 18,209.562 |
| total notebook seconds | 19,185.915 |

LightGBM train v1 completed:

| item | value |
| --- | --- |
| kernel | `kentookumura/exp075-compact-pfbeam-lgbm-train` v1 |
| accelerator | GPU (`enable_gpu=true`, `machine_shape=Gpu`) |
| feature kernel source | `kentookumura/exp075-compact-pfbeam-features-train` |
| mode | `gpu_repro_guard_dp_threads8` |
| CV RMSE `lgb0` | `9.76191961499488` |
| CV RMSE `lgb1` | `9.651408152898307` |
| CV RMSE `lgb2` | `9.65450091719246` |
| CV RMSE `lgb_mean` | `9.624618332949836` |
| `lgb_mean` prediction SHA | `4621598e334539194aaa60a448f32eaa446061c325ef433c11e11ffdb67cc846` |
| feature importance plot SHA | `f4798afebccdd470679cc16b8bf7bd9a14423c7be6f805b65e44027f28074740` |
| model manifest SHA | `39553e4342a1cf830f92ecf40eca7c218205359dce4008818965e908da9f2179` |
| metrics SHA | `f10749ce9549402b39e560813f273286833154706d74495524372ef08bebd6e7` |

Top mean feature importance:

| rank | feature | mean importance |
| --- | --- | ---: |
| 1 | `beam_vs_spatial` | 6210.533333 |
| 2 | `pf_vs_dense` | 5845.666667 |
| 3 | `pf_vs_spatial` | 5428.200000 |
| 4 | `beam_vcons_d` | 5352.266667 |
| 5 | `last_known_tvt` | 5181.266667 |

This train v1 section is retained as history only because it used pre-patch v3 features.

LightGBM train v2 completed:

| item | value |
| --- | --- |
| kernel | `kentookumura/exp075-compact-pfbeam-lgbm-train` v2 |
| URL | `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-lgbm-train` |
| accelerator | GPU (`enable_gpu=true`) |
| feature kernel source | `kentookumura/exp075-compact-pfbeam-features-train` v4 output |
| feature decompressed CSV content SHA | `047b80b32e64b595f2a75e7593ecb513e1f27d43de87614dd2de82dae416d5b4` |
| added analysis output | fold-mean feature importance CSV and matplotlib PNG |
| CV RMSE `lgb0` | `9.841188080569813` |
| CV RMSE `lgb1` | `9.727603469600522` |
| CV RMSE `lgb2` | `9.719506950985725` |
| CV RMSE `lgb_mean` | `9.699548082062895` |
| `lgb_mean` prediction SHA | `0afe646b8f52adbcea775a401c6d5af77e77df166bd4186d039ac438c9c46320` |
| fold-mean importance plot SHA | `78c6f0e36e0ee2970908d42a00f9058878b5fb4192bd9cf8bbd0988c608de7f6` |
| model manifest SHA | `ce4cf6897596c043a7a3a286a67607155092295f06715fdaf57539bba8fc1247` |
| metrics SHA | `e4d518ec27a8e94b13ddef5e5d40a853f5fc423c090efa9512280629be438b86` |
| elapsed seconds | `9484.825` |

Feature importance for v2 is computed by averaging LightGBM `feature_importances_` across configs within each fold, then averaging those per-fold values across folds. This is intended for exp075 vs exp076 LB analysis.

Top fold-mean feature importance for v2:

| rank | feature | mean importance |
| --- | --- | ---: |
| 1 | `beam_vs_spatial` | 8602.400000 |
| 2 | `pf_vs_dense` | 7885.800000 |
| 3 | `pf_vs_spatial` | 7475.600000 |
| 4 | `beam_vcons_d` | 6899.466667 |
| 5 | `pf_z_delta` | 6673.133333 |
| 6 | `last_known_tvt` | 6128.533333 |
| 7 | `beam_stiff_d` | 5872.800000 |
| 8 | `pf_vs_z` | 5689.333333 |
| 9 | `pf_ancc_delta` | 5514.666667 |
| 10 | `beam_vloose_d` | 5460.733333 |

Inference v1 completed:

| item | value |
| --- | --- |
| kernel | `kentookumura/exp075-compact-pfbeam-lgbm-infer` v1 |
| accelerator | GPU (`enable_gpu=true`, `machine_shape=Gpu`) |
| model kernel source | `kentookumura/exp075-compact-pfbeam-lgbm-train` |
| raw-test feature rows / wells / columns | 14,151 / 3 / 67 |
| raw-test feature decompressed CSV content SHA | `fa82323cff7d24712f109348313a47aaf965c8c8acaa8275f7224d33a35412e8` |
| raw-test feature raw gzip SHA | `1ad141560fe42b2021b46fef534464c1ab6cd8d53513d79c9452d5fc0a31b723` |
| prediction SHA | `03e5cf79fcf9e6f03e0725ea4acaa35b08b0748a1886f8ac10931b47f54cf07e` |
| submission SHA | `c962cbd1602511c973a7d92b6973c5db790ad7eab310649e025dff411a7c991e` |
| fallback rows | 0 |
| prediction range | 11599.796875 - 12240.2919921875 |
| submit-check | PASS |

Code submissions of the same inference v1 produced different Public LB scores:

| ref | submitted | Public LB | interpretation |
| --- | --- | ---: | --- |
| `53790771` | 2026-06-18 00:20:33.957000 | 8.535 | nonreproducible v1 draw |
| `53790878` | 2026-06-18 00:27:05.643000 | 8.447 | nonreproducible v1 draw; do not adopt as deterministic evidence |

## Interpretation

This implementation removed the duplicate train feature regeneration guard, but v1 did not actually use the established stable-seed PF/Beam generation method in the compact replay module. The hidden code-submit rerun exposed this: the same inference was submitted twice and received different Public LB scores.

Root cause: `public_notebook_replay_audit.py` used unseeded numba `np.random` inside `run_pf_ancc` / `run_pf_z`, and exp075 did not pass stable split/well seeds to likelihood-PF. A patch has been prepared by porting exp072/exp073's `stable_seed("pf_ancc", wid)`, `stable_seed("pf_z", wid)`, and `stable_seed("likpf", split, wid)` policy into exp075.

The compact 65-feature surface should be treated as reproducibility-guarded from the patched v4/v2/v3 chain, not from the pre-patch v3/v1/v1 chain.

Inference v2 CPU and v3 GPU completed from train v2:

| item | CPU v2 | GPU v3 |
| --- | --- | --- |
| model source | train v2 | train v2 |
| accelerator | CPU (`enable_gpu=false`) | GPU (`enable_gpu=true`) |
| raw-test feature content SHA | `38512547d23528e713134493311dae26707d30c9b302958cb8c6ff3ce02bb0a2` | `6a00cb045dc1bdd3e8627bd669b445a97790ff7234054c3968225be62d49401d` |
| prediction SHA | `137baa962a02ca6b2d63c6e1085250ccb3b2ec158e098278bafc04799291bf51` | `137baa962a02ca6b2d63c6e1085250ccb3b2ec158e098278bafc04799291bf51` |
| submission SHA | `79da931b3dd651fd7cc983d0b90de4d298995b8d68d1665a8a48edf621725284` | `79da931b3dd651fd7cc983d0b90de4d298995b8d68d1665a8a48edf621725284` |
| submit-check | PASS | PASS |
| elapsed seconds | `149.15` | `126.607` |

The final prediction and submission are stable between CPU v2 and GPU v3, but the regenerated test feature content SHA differs by runtime mode. Strict feature-content reproducibility comparisons should therefore compare reruns under the same runtime mode.

GPU inference v3 code submissions:

| ref | submitted | Public LB | interpretation |
| --- | --- | ---: | --- |
| `53807892` | 2026-06-18 12:21:41.770000 | 8.489 | valid exp075 GPU inference v3 submission |
| `53807896` | 2026-06-18 12:22:00.443000 | 8.489 | duplicate exp075 GPU inference v3 submission with matching Public LB |

The latest observed submission `ref=53809333` is not related to this experiment per user clarification and is not attributed to exp075.

## Next

Use exp075 as the reproducibility-guarded compact PF/Beam surface for exp070/exp074 follow-up analysis. For strict feature-content equality, rerun under the same runtime mode and compare decompressed CSV content SHA, prediction SHA, and submission SHA.
