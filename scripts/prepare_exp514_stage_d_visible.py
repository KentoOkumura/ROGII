from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512"
BASE_SOURCE = (
    EXP_DIR
    / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py"
)
OUTPUT_SOURCE = (
    EXP_DIR / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_d_visible.py"
)

EXPECTED_BASE_SOURCE_SHA256 = (
    "961762731f91bf20de6d43d869aeed44bfa98f60be7f8cccc1c65b37d05dc24c"
)
EXPECTED_BASE_GENERATOR_SHA256 = (
    "909485a09c8224a41f9b088452b2c1a7c8f45e66273daaea8871d55b3250ba26"
)
EXPECTED_STAGE_D_V1_HJYACT_SHA256 = (
    "6b3e1c576afc47f065bdcce12a09f4361a6bb97c63667630f4f5ab1e64fa37b3"
)
EXPECTED_STAGE_D_V2_GOLD_BALANCED_SHA256 = (
    "2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815"
)
EXPECTED_STAGE_D_V2_EXP413_COMPONENT_SHA256 = (
    "04e6da90cee4325fb01bf7ce49bd87b91b16cf675cfb9d4cdaec77904aee5908"
)
EXPECTED_STAGE_D_V2_COMPONENT_READOUT_SHA256 = (
    "c3a9b217568fdd8d09eea337e9d1d5addb9f6c6b26b138d7865caee1ffe7e1fd"
)
EXPECTED_STAGE_D_V2_FINAL_SUBMISSION_SHA256 = (
    "9974c3face9004ffb39ead3c6d8955dff5d540559c48cd45bc3fbaebf2e192ad"
)


PARENT_VISIBLE_HJYACT_GUARD = '''visible_reference_checks = {"sample_id_order_match": sample_id_sha == VISIBLE_SAMPLE_ID_ORDER_SHA256}
if visible_reference_checks["sample_id_order_match"]:
    observed_hjyact_sha = sha256_file(source_submission_path)
    visible_reference_checks["hjyact_submission_sha256"] = observed_hjyact_sha
    visible_reference_checks["hjyact_submission_match"] = observed_hjyact_sha == SOURCE_VISIBLE_FINAL_SHA256
    if not visible_reference_checks["hjyact_submission_match"]:
        raise RuntimeError(
            f"visible hjyact-v2 parity failed: {observed_hjyact_sha} != {SOURCE_VISIBLE_FINAL_SHA256}"
        )
else:
    visible_reference_checks["hjyact_submission_match"] = None
'''


STAGE_D_CANDIDATE_VISIBLE_HJYACT_GUARD = '''visible_reference_checks = {"sample_id_order_match": sample_id_sha == VISIBLE_SAMPLE_ID_ORDER_SHA256}
visible_reference_checks["hjyact_parent_exact_match_required"] = False
if visible_reference_checks["sample_id_order_match"]:
    observed_hjyact_sha = sha256_file(source_submission_path)
    visible_reference_checks["hjyact_submission_sha256"] = observed_hjyact_sha
    visible_reference_checks["hjyact_parent_submission_match"] = (
        observed_hjyact_sha == SOURCE_VISIBLE_FINAL_SHA256
    )
    visible_reference_checks["hjyact_candidate_submission_match"] = (
        observed_hjyact_sha == STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256
    )
    visible_reference_checks["hjyact_submission_match"] = visible_reference_checks[
        "hjyact_candidate_submission_match"
    ]
    if not visible_reference_checks["hjyact_candidate_submission_match"]:
        raise RuntimeError(
            "visible exp514 HJYACT candidate witness failed: "
            f"{observed_hjyact_sha} != {STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256}"
        )
else:
    visible_reference_checks["hjyact_parent_submission_match"] = None
    visible_reference_checks["hjyact_candidate_submission_match"] = None
    visible_reference_checks["hjyact_submission_match"] = None
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one replacement target, found {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def apply_memory_lifetime_patches(source: str) -> str:
    source = replace_once(
        source,
        '''def shared_likpf_sp45_adapter(record):
    ledger = record["ledger"]
    ledger["sp45_consumer_hits"] += 1
    if ledger["sp45_consumer_hits"] != 1:
        raise RuntimeError(f"SP45 consumed shared bank more than once: {record['well']}")
    required = {f"pf_scale_{scale:g}" for scale in SHARED_LIKPF_SCALES}
    required.add("pf_mean")
    if set(record["sp45_full"]) != required:
        raise ValueError(f"SP45 shared adapter schema mismatch: {record['well']}")
    return (
        {name: values.copy() for name, values in record["sp45_full"].items()},
        dict(record["branch_summary"]),
    )


def shared_likpf_exp413_adapter(bank, wells):
    frames = []
    for well in [str(value) for value in wells]:
        if well not in bank:
            raise KeyError(f"exp413 shared likelihood-PF bank is missing well {well}")
        record = bank[well]
        record["ledger"]["exp413_consumer_hits"] += 1
        if record["ledger"]["exp413_consumer_hits"] != 1:
            raise RuntimeError(f"exp413 consumed shared bank more than once: {well}")
        frames.append(record["exp413_frame"].copy(deep=True))
    frame = pd.concat(frames, ignore_index=True)
    expected_columns = ["id", "likpf_scale_5", "likpf_mean"]
    if list(frame.columns) != expected_columns or frame["id"].duplicated().any():
        raise ValueError("exp413 shared likelihood-PF adapter schema/ID contract failed")
    return frame
''',
        '''def shared_likpf_sp45_adapter(record):
    ledger = record["ledger"]
    ledger["sp45_consumer_hits"] += 1
    if ledger["sp45_consumer_hits"] != 1:
        raise RuntimeError(f"SP45 consumed shared bank more than once: {record['well']}")
    required = {f"pf_scale_{scale:g}" for scale in SHARED_LIKPF_SCALES}
    required.add("pf_mean")
    if set(record["sp45_full"]) != required:
        raise ValueError(f"SP45 shared adapter schema mismatch: {record['well']}")
    # Transfer array ownership to the one SP45 consumer instead of duplicating
    # every full-length scale. The worker drops record['sp45_full'] immediately
    # after the selector returns.
    return dict(record["sp45_full"]), dict(record["branch_summary"])


def release_shared_likpf_sp45_payload(record):
    released = []
    for key in ("sp45_full", "row_index", "evaluation_index", "known_mask"):
        if key in record:
            record.pop(key)
            released.append(key)
    record["audit"]["sp45_payload_released"] = True
    record["audit"]["released_after_sp45"] = released
    record["audit"]["retained_for_exp413"] = [
        "id", "likpf_scale_5", "likpf_mean"
    ]
    return released


def shared_likpf_exp413_adapter(bank, wells):
    frames = []
    records = []
    for well in [str(value) for value in wells]:
        if well not in bank:
            raise KeyError(f"exp413 shared likelihood-PF bank is missing well {well}")
        record = bank[well]
        record["ledger"]["exp413_consumer_hits"] += 1
        if record["ledger"]["exp413_consumer_hits"] != 1:
            raise RuntimeError(f"exp413 consumed shared bank more than once: {well}")
        frames.append(record["exp413_frame"])
        records.append(record)
    frame = pd.concat(frames, ignore_index=True)
    expected_columns = ["id", "likpf_scale_5", "likpf_mean"]
    if list(frame.columns) != expected_columns or frame["id"].duplicated().any():
        raise ValueError("exp413 shared likelihood-PF adapter schema/ID contract failed")
    for record in records:
        record.pop("exp413_frame")
        record["audit"]["exp413_payload_released"] = True
    return frame
''',
    )
    source = replace_once(
        source,
        '''        if any(key in record for key in ("predictions", "log_likelihoods", "raw_bank")):
            raise RuntimeError(f"raw likelihood-PF bank leaked beyond well scope: {well}")
''',
        '''        forbidden_payloads = {
            "predictions", "log_likelihoods", "raw_bank", "sp45_full",
            "row_index", "evaluation_index", "known_mask", "exp413_frame",
        }
        leaked_payloads = sorted(forbidden_payloads.intersection(record))
        if leaked_payloads:
            raise RuntimeError(
                f"shared likelihood-PF payload leaked beyond consumer scope: "
                f"{well}: {leaked_payloads}"
            )
''',
    )
    source = replace_once(
        source,
        '''test_df2 = test_df.copy()
pf_test = test_df2['pf_ancc'].values - test_df2['last_known_tvt'].values

test_df2['pred'] = test_df2['last_known_tvt'].values + apply_pp(
    test_df2, 
    ridge_test_preds,
    pf_test, 
    **pp_params
)
test_df2 = sg_smooth(test_df2, 'pred')
''',
        '''pf_test = test_df['pf_ancc'].values - test_df['last_known_tvt'].values
test_df2 = test_df[['id', 'well', 'md_since']].copy()
test_df2['pred'] = test_df['last_known_tvt'].values + apply_pp(
    test_df,
    ridge_test_preds,
    pf_test,
    **pp_params
)
test_df2 = sg_smooth(test_df2, 'pred')
''',
    )
    source = replace_once(
        source,
        '''sub_1['tvt']=sub_1['tvt'].fillna(float(train_df['last_known_tvt'].mean()+train_df['target'].mean()))
sub_1
''',
        '''sub_1['tvt']=sub_1['tvt'].fillna(float(train_df['last_known_tvt'].mean()+train_df['target'].mean()))

# The Ridge prediction is now fully represented by sub_1. Release train-side
# frames, OOF arrays, saved trainer wrappers, and redundant test aliases before
# the 128-seed shared PF starts.
RIDGE_MEMORY_RELEASE_REPORT = _exp514_release_globals(
    (
        '_ridge_train_path', 'train_df', 'test_paths', 'test_df', 'features',
        'X', 'y', 'g', 'X_test', 'oof_preds', 'test_preds', 'trainer',
        'ridge_trainer', 'ridge_oof_preds', 'ridge_test_preds', 'overall_scores',
        'fold_scores', 'base', 'ytrue', 'pf_oof', 'pf_test', 'd', 'ridge_score',
        'test_df2', 'sample_sub', 'ridge_params', 'pp_params',
    ),
    label='ridge_train_and_prediction_intermediates',
)
sub_1
''',
    )
    source = replace_once(
        source,
        '''if not test_wells:
    raise FileNotFoundError('shared likelihood-PF requires at least one dynamic test well')
SHARED_LIKPF_BANK, SHARED_LIKPF_PARALLEL_REPORT = materialize_shared_likpf_bank(
    test_wells,
    'test',
    load_well,
    n_jobs=SHARED_LIKPF_N_JOBS,
)
print('shared likelihood-PF producer report:', SHARED_LIKPF_PARALLEL_REPORT)
''',
        '''if not test_wells:
    raise FileNotFoundError('shared likelihood-PF requires at least one dynamic test well')
''',
    )
    source = replace_once(
        source,
        '''_sp45_parallel_started = time.time()


def _run_sp45_test_well(order, wid):
''',
        '''_shared_sp45_pipeline_started = time.time()


def _run_sp45_test_well(order, wid, shared_record):
''',
    )
    source = replace_once(
        source,
        '''    pf_by_scale, _seed_branch = shared_likpf_sp45_adapter(
        SHARED_LIKPF_BANK[str(wid)]
    )
''',
        '''    pf_by_scale, _seed_branch = shared_likpf_sp45_adapter(shared_record)
''',
    )
    source = replace_once(
        source,
        '''_sp45_well_results = Parallel(
    n_jobs=_sp45_effective_n_jobs,
    backend='threading',
)(
    delayed(_run_sp45_test_well)(order, wid)
    for order, wid in enumerate(test_wells)
)
for _well_result in _sp45_well_results:
    for _message in _well_result['messages']:
        print(_message)
    rows.extend(_well_result['rows'])
    bimodal_report_rows.append(_well_result['bimodal_report'])
    if _well_result['seed_branch']:
        PF_SEED_BRANCH_STATS[_well_result['well']] = _well_result['seed_branch']
SP45_WELL_PARALLEL_REPORT = {
    'requested_n_jobs': SP45_WELL_N_JOBS,
    'effective_n_jobs': _sp45_effective_n_jobs,
    'test_wells': len(test_wells),
    'backend': 'threading',
    'elapsed_seconds': round(time.time() - _sp45_parallel_started, 3),
}
print('SP45 well-parallel report:', SP45_WELL_PARALLEL_REPORT)
''',
        '''def _run_shared_likpf_sp45_well(order, wid):
    shared_started = time.time()
    record = _shared_likpf_one_well(
        str(wid),
        'test',
        load_well,
        particles=SHARED_LIKPF_PARTICLES,
        seeds=SHARED_LIKPF_SEEDS,
    )
    shared_seconds = time.time() - shared_started
    sp45_started = time.time()
    well_result = _run_sp45_test_well(order, wid, record)
    sp45_seconds = time.time() - sp45_started
    released_payloads = release_shared_likpf_sp45_payload(record)
    return {
        'order': int(order),
        'well': str(wid),
        'record': record,
        'well_result': well_result,
        'shared_seconds': float(shared_seconds),
        'sp45_seconds': float(sp45_seconds),
        'released_payloads': released_payloads,
    }


_shared_sp45_pipeline_results = Parallel(
    n_jobs=_sp45_effective_n_jobs,
    backend='threading',
)(
    delayed(_run_shared_likpf_sp45_well)(order, wid)
    for order, wid in enumerate(test_wells)
)
_shared_sp45_pipeline_results = sorted(
    _shared_sp45_pipeline_results, key=lambda item: int(item['order'])
)
if [item['well'] for item in _shared_sp45_pipeline_results] != [
    str(well) for well in test_wells
]:
    raise RuntimeError('shared PF/SP45 streaming pipeline changed well order')

SHARED_LIKPF_BANK = {}
for _pipeline_result in _shared_sp45_pipeline_results:
    _record = _pipeline_result['record']
    _well = _pipeline_result['well']
    if any(
        key in _record
        for key in ('sp45_full', 'row_index', 'evaluation_index', 'known_mask')
    ):
        raise RuntimeError(f'shared PF full payload retained after SP45: {_well}')
    SHARED_LIKPF_BANK[_well] = _record
    _well_result = _pipeline_result['well_result']
    for _message in _well_result['messages']:
        print(_message)
    rows.extend(_well_result['rows'])
    bimodal_report_rows.append(_well_result['bimodal_report'])
    if _well_result['seed_branch']:
        PF_SEED_BRANCH_STATS[_well_result['well']] = _well_result['seed_branch']

_shared_sp45_pipeline_elapsed = time.time() - _shared_sp45_pipeline_started
_shared_worker_seconds = float(
    sum(item['shared_seconds'] for item in _shared_sp45_pipeline_results)
)
_sp45_worker_seconds = float(
    sum(item['sp45_seconds'] for item in _shared_sp45_pipeline_results)
)
_combined_worker_seconds = _shared_worker_seconds + _sp45_worker_seconds
_shared_elapsed_share = (
    _shared_sp45_pipeline_elapsed * _shared_worker_seconds / _combined_worker_seconds
    if _combined_worker_seconds > 0.0
    else 0.0
)
_sp45_elapsed_share = _shared_sp45_pipeline_elapsed - _shared_elapsed_share
SHARED_LIKPF_PARALLEL_REPORT = {
    'requested_n_jobs': SHARED_LIKPF_N_JOBS,
    'effective_n_jobs': _sp45_effective_n_jobs,
    'backend': 'threading',
    'wells': len(test_wells),
    'worker_seconds_sum': round(_shared_worker_seconds, 6),
    'elapsed_seconds': round(_shared_elapsed_share, 6),
    'elapsed_semantics': 'proportional_share_of_streaming_pipeline_wall_time',
    'all_well_full_payload_retained': False,
    'max_concurrent_full_payload_wells': _sp45_effective_n_jobs,
}
SP45_WELL_PARALLEL_REPORT = {
    'requested_n_jobs': SP45_WELL_N_JOBS,
    'effective_n_jobs': _sp45_effective_n_jobs,
    'test_wells': len(test_wells),
    'backend': 'threading',
    'worker_seconds_sum': round(_sp45_worker_seconds, 6),
    'elapsed_seconds': round(_sp45_elapsed_share, 6),
    'elapsed_semantics': 'proportional_share_of_streaming_pipeline_wall_time',
}
SHARED_SP45_STREAMING_REPORT = {
    'requested_n_jobs': SHARED_LIKPF_N_JOBS,
    'effective_n_jobs': _sp45_effective_n_jobs,
    'backend': 'threading',
    'wells': len(test_wells),
    'elapsed_seconds': round(_shared_sp45_pipeline_elapsed, 6),
    'full_payload_retention': 'at_most_effective_n_jobs_then_release_after_sp45',
    'retained_for_exp413': ['id', 'likpf_scale_5', 'likpf_mean'],
}
del _shared_sp45_pipeline_results
_shared_sp45_gc_collected = int(_exp514_gc.collect())
SHARED_SP45_STREAMING_REPORT['post_pipeline_gc_collected'] = _shared_sp45_gc_collected
print('shared likelihood-PF producer report:', SHARED_LIKPF_PARALLEL_REPORT)
print('SP45 well-parallel report:', SP45_WELL_PARALLEL_REPORT)
print('shared PF/SP45 streaming report:', SHARED_SP45_STREAMING_REPORT)
''',
    )
    source = replace_once(
        source,
        '''def add_likpf_features(df, likpf):
    df = df.merge(likpf, on="id", how="left")
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = df[c].fillna(df["last_known_tvt"]); df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df
''',
        '''def add_likpf_features(df, likpf):
    if df["id"].duplicated().any() or likpf["id"].duplicated().any():
        raise ValueError("likelihood-PF feature alignment requires unique IDs")
    aligned = (
        likpf.assign(id=likpf["id"].astype(str))
        .set_index("id")
        .reindex(df["id"].astype(str))
    )
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = aligned[c].to_numpy(copy=False)
        df[c] = df[c].fillna(df["last_known_tvt"])
        df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df
''',
    )
    source = replace_once(
        source,
        '''    pf_frame = pd.concat(pf_blocks, ignore_index=True)
    if pf_frame["id"].duplicated().any() or len(pf_frame) != len(base):
        raise ValueError("HJYACT PF-only refresh ID contract failed")
    result = base.copy(deep=True).reset_index(drop=True)
''',
        '''    pf_frame = pd.concat(pf_blocks, ignore_index=True)
    del pf_blocks, by_well
    _exp514_gc.collect()
    if pf_frame["id"].duplicated().any() or len(pf_frame) != len(base):
        raise ValueError("HJYACT PF-only refresh ID contract failed")
    # SP45 has finished consuming this frame. Move its sole ownership into
    # HJYACT and update only the stochastic PF-dependent columns in place.
    result = base
    result.reset_index(drop=True, inplace=True)
''',
    )
    source = replace_once(
        source,
        '''    result = result.drop(columns=list(HJYACT_AUXILIARY_SHARED_COLUMNS))
    reused_columns = [
''',
        '''    result.drop(columns=list(HJYACT_AUXILIARY_SHARED_COLUMNS), inplace=True)
    reused_columns = [
''',
    )
    source = replace_once(
        source,
        '''        "auxiliary_columns_dropped": list(HJYACT_AUXILIARY_SHARED_COLUMNS),
        "elapsed_pf_refresh_seconds": round(time.time() - started, 6),
    }
    return result
''',
        '''        "auxiliary_columns_dropped": list(HJYACT_AUXILIARY_SHARED_COLUMNS),
        "ownership_transfer": "sp45_to_hjyact_in_place",
        "full_frame_deep_copy_count": 0,
        "elapsed_pf_refresh_seconds": round(time.time() - started, 6),
    }
    globals()["SP45_SHARED_TEST_FEATURE_FRAME"] = None
    return result
''',
    )
    source = replace_once(
        source,
        '''    shared = build_hjyact_features_from_sp45(test_wids, "test")
    globals()["HJYACT_SHARED_FEATURE_RUNTIME_SECONDS"] = time.time() - shared_started
    test_df = add_likpf_features(shared, likpf_test).reset_index(drop=True)
''',
        '''    shared = build_hjyact_features_from_sp45(test_wids, "test")
    globals()["HJYACT_SHARED_FEATURE_RUNTIME_SECONDS"] = time.time() - shared_started
    test_df = add_likpf_features(shared, likpf_test)
    test_df.reset_index(drop=True, inplace=True)
    del likpf_test
    _exp514_gc.collect()
''',
    )
    source = replace_once(
        source,
        '''    pf_frame = shared_deterministic_frame.copy(deep=True).reset_index(drop=True)
    pf_frame["id"] = pf_frame["id"].astype(str)
''',
        '''    # The caller transfers sole ownership with globals().pop(). Reuse the
    # frame directly; the HJYACT component and reuse SHA records are already frozen.
    pf_frame = shared_deterministic_frame
    shared_deterministic_frame = None
    pf_frame.reset_index(drop=True, inplace=True)
    pf_frame["id"] = pf_frame["id"].astype(str)
''',
    )
    source = replace_once(
        source,
        '''    likpf_columns = [column for column in pf_frame if column.startswith("likpf_")]
    pf_frame = pf_frame.drop(columns=likpf_columns)
''',
        '''    likpf_columns = [column for column in pf_frame if column.startswith("likpf_")]
    pf_frame.drop(columns=likpf_columns, inplace=True)
''',
    )
    source = replace_once(
        source,
        '''    likpf_test = shared_likpf_exp413_adapter(shared_likpf_bank, test_wells)
    pf_frame = pf_frame.merge(likpf_test, on='id', how='left', validate='one_to_one')
    for likpf_column in [column for column in likpf_test.columns if column != 'id']:
        if pf_frame[likpf_column].isna().any():
            raise ValueError(f'shared exp413 likelihood-PF coverage failed: {likpf_column}')
        pf_frame[likpf_column] = pf_frame[likpf_column].astype(np.float32)
        pf_frame[likpf_column + '_d'] = (
            pf_frame[likpf_column] - pf_frame['last_known_tvt']
        ).astype(np.float32)
    pf_frame = pf_frame.reset_index(drop=True)
''',
        '''    likpf_test = shared_likpf_exp413_adapter(shared_likpf_bank, test_wells)
    _likpf_test_rows = len(likpf_test)
    _likpf_aligned = (
        likpf_test.assign(id=likpf_test['id'].astype(str))
        .set_index('id')
        .reindex(pf_frame['id'].astype(str))
    )
    for likpf_column in [column for column in likpf_test.columns if column != 'id']:
        if _likpf_aligned[likpf_column].isna().any():
            raise ValueError(f'shared exp413 likelihood-PF coverage failed: {likpf_column}')
        pf_frame[likpf_column] = _likpf_aligned[likpf_column].to_numpy(
            dtype=np.float32, copy=False
        )
        pf_frame[likpf_column + '_d'] = (
            pf_frame[likpf_column] - pf_frame['last_known_tvt']
        ).astype(np.float32)
    del likpf_test, _likpf_aligned
    _exp514_gc.collect()
    pf_frame.reset_index(drop=True, inplace=True)
''',
    )
    source = replace_once(
        source,
        '''        reuse_tracker.mark_exp413_hit(route_pf_result["well"])
        route_pf_records.append(route_pf_result["record"])

    if shared_likpf_bank is None:
''',
        '''        reuse_tracker.mark_exp413_hit(route_pf_result["well"])
        route_pf_records.append(route_pf_result["record"])

    # All route-PF updates have been applied to pf_frame. Drop the per-well
    # result list and ID lookup before materializing the compact likPF columns.
    route_pf_results = None
    route_pf_result = None
    id_to_position = None
    positions = None
    values = None
    _exp514_gc.collect()

    if shared_likpf_bank is None:
''',
    )
    source = replace_once(
        source,
        '''        "test_likpf_rows": len(likpf_test),
''',
        '''        "test_likpf_rows": int(_likpf_test_rows),
''',
    )
    source = replace_once(
        source,
        '''    learned_source_frame = exp145.ensure_candidate_value_columns(
        pf_frame.copy(), exp145_candidates
    )
    learned_cache_path = output_dir / "exp263_replay_for_exp145.csv.gz"
    learned_source_frame.to_csv(learned_cache_path, index=False, compression="gzip")
''',
        '''    learned_source_frame = exp145.ensure_candidate_value_columns(
        pf_frame, exp145_candidates
    )
    learned_cache_path = output_dir / "exp263_replay_for_exp145.csv.gz"
    learned_source_frame.to_csv(learned_cache_path, index=False, compression="gzip")
    del learned_source_frame
    _exp514_gc.collect()
''',
    )
    source = replace_once(
        source,
        '''    test_frame, anchor_meta = exp218.add_inference_anchor_columns(
        pf_frame.copy(), paths.test_data_dir
    )
''',
        '''    test_frame, anchor_meta = exp218.add_inference_anchor_columns(
        pf_frame, paths.test_data_dir
    )
    pf_frame = None
    _exp514_gc.collect()
''',
    )
    source = replace_once(
        source,
        '''    del projection, learned_source, learned, grwr, learned_source_frame
''',
        '''    del projection, learned_source, learned, grwr
''',
    )
    source = replace_once(
        source,
        '''    return predictions.copy(), dict(metrics), Path(prediction_path)
''',
        '''    return int(len(predictions)), dict(metrics), Path(prediction_path)
''',
    )
    source = replace_once(
        source,
        '''exp413_predictions_memory, exp413_metrics, exp413_prediction_path = generate_dynamic_exp413_prediction(
    shared_deterministic_frame=HJYACT_SHARED_FEATURE_FRAME,
    reuse_tracker=CANDIDATE_REUSE_TRACKER,
    shared_likpf_bank=SHARED_LIKPF_BANK,
)
''',
        '''HJYACT_SHARED_FEATURE_WELL_COUNT = int(
    HJYACT_SHARED_FEATURE_FRAME["well"].nunique()
)
EXP413_PRE_RELEASE_REPORT = _exp514_release_globals(
    (
        "PF_SEED_BRANCH_STATS", "rows", "bimodal_report_rows", "sub_1",
        "sub_2", "sub", "cv_final", "_bimodal_df", "_active_mask",
        "_active_cols", "train_hw_files", "test_hw_files",
    ),
    label="consumed_sp45_and_hjyact_intermediates_before_exp413",
)
exp413_prediction_rows_memory, exp413_metrics, exp413_prediction_path = generate_dynamic_exp413_prediction(
    shared_deterministic_frame=globals().pop("HJYACT_SHARED_FEATURE_FRAME"),
    reuse_tracker=CANDIDATE_REUSE_TRACKER,
    shared_likpf_bank=SHARED_LIKPF_BANK,
)
''',
    )
    source = replace_once(
        source,
        '''if len(exp413_predictions_memory) != len(exp413_predictions):
''',
        '''if exp413_prediction_rows_memory != len(exp413_predictions):
''',
    )
    source = replace_once(
        source,
        '''shared_likpf_manifest = finalize_shared_likpf_manifest(
    SHARED_LIKPF_BANK,
    test_wells,
)
shared_likpf_manifest['parallel_report'] = SHARED_LIKPF_PARALLEL_REPORT
''',
        '''shared_likpf_manifest = finalize_shared_likpf_manifest(
    SHARED_LIKPF_BANK,
    test_wells,
)
SHARED_BANK_FINAL_RELEASE_REPORT = _exp514_release_globals(
    ("SHARED_LIKPF_BANK",),
    label="consumed_shared_likpf_manifest_records",
)
shared_likpf_manifest['parallel_report'] = SHARED_LIKPF_PARALLEL_REPORT
''',
    )
    source = replace_once(
        source,
        '''    "wells": int(HJYACT_SHARED_FEATURE_FRAME["well"].nunique()),
''',
        '''    "wells": HJYACT_SHARED_FEATURE_WELL_COUNT,
''',
    )
    source = replace_once(
        source,
        '''    "runtime_optimizations": {
        "sp45_well_parallel": SP45_WELL_PARALLEL_REPORT,
        "exp413_well_n_jobs": EXP413_WELL_N_JOBS,
        "model_package_correction_enabled": RUN_MODEL_PACKAGE_CORRECTION,
    },
''',
        '''    "runtime_optimizations": {
        "sp45_well_parallel": SP45_WELL_PARALLEL_REPORT,
        "shared_likpf_sp45_streaming": SHARED_SP45_STREAMING_REPORT,
        "ridge_memory_release": RIDGE_MEMORY_RELEASE_REPORT,
        "pre_exp413_memory_release": EXP413_PRE_RELEASE_REPORT,
        "shared_bank_final_release": SHARED_BANK_FINAL_RELEASE_REPORT,
        "dataframe_ownership_transfer": {
            "sp45_to_hjyact": True,
            "hjyact_to_exp413": True,
            "exp413_returned_prediction_frame": False,
        },
        "exp413_well_n_jobs": EXP413_WELL_N_JOBS,
        "model_package_correction_enabled": RUN_MODEL_PACKAGE_CORRECTION,
    },
''',
    )
    return source


RUNTIME_REPORT_CELL = r'''

# %% [markdown]
# ## 10. Stage D visible runtime and 200-well estimate

# %%
import resource as _stage_d_resource

_stage_d_visible_wells = int(len(test_wells))
_stage_d_visible_rows = int(len(submission))
if _stage_d_visible_wells < 1 or _stage_d_visible_rows < 1:
    raise RuntimeError("Stage D visible runtime report requires nonempty wells and rows")

_stage_d_gold_path = WORKING_DIR / "gold_prefix_submission_audit.json"
if bool(_GOLD_ENABLE) and not _stage_d_gold_path.is_file():
    raise RuntimeError("Stage D enabled Gold stage did not write its runtime audit")
_stage_d_gold_seconds = 0.0
if _stage_d_gold_path.is_file():
    _stage_d_gold_seconds = float(
        json.loads(_stage_d_gold_path.read_text(encoding="utf-8"))["elapsed_sec"]
    )

if HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS is None:
    raise RuntimeError("Stage D learned trajectory total runtime was not recorded")

_stage_d_v2_equivalence_path = (
    WORKING_DIR / "exp514_stage_d_v2_output_equivalence.json"
)
if visible_reference_checks["sample_id_order_match"]:
    _stage_d_v2_equivalence_targets = {
        "gold_balanced": (
            WORKING_DIR / "submission_gold_prefix_balanced.csv",
            STAGE_D_V2_GOLD_BALANCED_SHA256,
        ),
        "hjyact_component": (
            HJYACT_COMPONENT_PATH,
            STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256,
        ),
        "exp413_component": (
            EXP413_COMPONENT_PATH,
            STAGE_D_V2_EXP413_COMPONENT_SHA256,
        ),
        "component_readout": (
            WORKING_DIR / "exp514_component_readout.csv",
            STAGE_D_V2_COMPONENT_READOUT_SHA256,
        ),
        "final_submission": (
            FINAL_SUBMISSION_PATH,
            STAGE_D_V2_FINAL_SUBMISSION_SHA256,
        ),
    }
    _stage_d_v2_equivalence = {}
    for _stage_d_name, (_stage_d_path, _stage_d_expected_sha) in (
        _stage_d_v2_equivalence_targets.items()
    ):
        if not _stage_d_path.is_file():
            raise FileNotFoundError(
                f"Stage D v2 equivalence target missing: {_stage_d_name}: {_stage_d_path}"
            )
        _stage_d_observed_sha = sha256_file(_stage_d_path)
        _stage_d_match = _stage_d_observed_sha == _stage_d_expected_sha
        _stage_d_v2_equivalence[_stage_d_name] = {
            "path": str(_stage_d_path),
            "expected_sha256": _stage_d_expected_sha,
            "observed_sha256": _stage_d_observed_sha,
            "exact_match": bool(_stage_d_match),
        }
        if not _stage_d_match:
            raise RuntimeError(
                "Stage D v4 runtime-only output parity failed for "
                f"{_stage_d_name}: {_stage_d_observed_sha} != {_stage_d_expected_sha}"
            )
    _stage_d_v2_equivalence_manifest = {
        "reference_kernel": "kentookumura/exp514-shared-likpf-stage-d-visible",
        "reference_kernel_version": 2,
        "status": "PASS",
        "all_exact": True,
        "targets": _stage_d_v2_equivalence,
    }
else:
    _stage_d_v2_equivalence_manifest = {
        "reference_kernel": "kentookumura/exp514-shared-likpf-stage-d-visible",
        "reference_kernel_version": 2,
        "status": "SKIPPED_HIDDEN_DYNAMIC",
        "all_exact": None,
        "targets": {},
        "reason": "visible output SHA witnesses do not apply to hidden dynamic IDs",
    }
_stage_d_v2_equivalence_path.write_text(
    json.dumps(
        _stage_d_v2_equivalence_manifest,
        indent=2,
        sort_keys=True,
        default=str,
    )
    + "\n",
    encoding="utf-8",
)

_stage_d_parallel_stages = [
    {
        "name": "shared_likpf_sp45_streaming_pipeline",
        "seconds": float(SHARED_SP45_STREAMING_REPORT["elapsed_seconds"]),
        "scaling": "parallel_4",
    },
    {
        "name": "exp413_full_after_shared_pf",
        "seconds": float(exp413_metrics["runtime_seconds"]),
        "scaling": "parallel_4_mixed",
    },
    {
        "name": "gold_visible_prefix",
        "seconds": float(_stage_d_gold_seconds),
        "scaling": "parallel_4_process",
    },
]
_stage_d_sequential_stages = [
    {
        "name": "hjyact_learned_trajectory_total",
        "seconds": float(HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS),
        "scaling": "sequential_visible_throughput_conservative",
    },
]
_stage_d_all_stages = _stage_d_parallel_stages + _stage_d_sequential_stages
if any(item["seconds"] < 0.0 for item in _stage_d_all_stages):
    raise RuntimeError("Stage D stage runtime cannot be negative")

_stage_d_started = float(
    globals().get("_KAGGLE_BOOTSTRAP_STARTED", STAGE_D_VISIBLE_STARTED)
)
_stage_d_total_seconds = float(time.time() - _stage_d_started)
_stage_d_known_seconds = float(sum(item["seconds"] for item in _stage_d_all_stages))
_stage_d_fixed_overhead_seconds = float(max(0.0, _stage_d_total_seconds - _stage_d_known_seconds))
_stage_d_target_wells = 200
_stage_d_parallel_workers = 4

_stage_d_lower_seconds = _stage_d_fixed_overhead_seconds
_stage_d_upper_seconds = _stage_d_fixed_overhead_seconds
for _stage_d_item in _stage_d_parallel_stages:
    _stage_d_lower_seconds += (
        _stage_d_item["seconds"] * _stage_d_target_wells / _stage_d_parallel_workers
    )
    _stage_d_upper_seconds += (
        _stage_d_item["seconds"] * _stage_d_target_wells / _stage_d_visible_wells
    )
for _stage_d_item in _stage_d_sequential_stages:
    _stage_d_scaled = (
        _stage_d_item["seconds"] * _stage_d_target_wells / _stage_d_visible_wells
    )
    _stage_d_lower_seconds += _stage_d_scaled
    _stage_d_upper_seconds += _stage_d_scaled

_stage_d_peak_rss_mib = float(
    _stage_d_resource.getrusage(_stage_d_resource.RUSAGE_SELF).ru_maxrss / 1024.0
)
_stage_d_estimated_pass = bool(_stage_d_upper_seconds <= 32400.0)
_stage_d_report = {
    "experiment": EXPERIMENT_NAME,
    "stage": "stage_d_submission_ready_visible_test",
    "status": "PASS" if FINAL_SUBMISSION_PATH.is_file() else "FAIL",
    "runtime_estimate_status": (
        "estimated_pass_not_hidden_runtime_guarantee"
        if _stage_d_estimated_pass
        else "estimated_fail"
    ),
    "base_candidate_source_sha256": STAGE_D_BASE_CANDIDATE_SHA256,
    "generator_sha256": STAGE_D_GENERATOR_SHA256,
    "visible": {
        "wells": _stage_d_visible_wells,
        "rows": _stage_d_visible_rows,
        "total_seconds_including_bootstrap": round(_stage_d_total_seconds, 6),
        "peak_parent_process_rss_mib": round(_stage_d_peak_rss_mib, 3),
        "peak_rss_scope": "parent_process_only_excludes_child_subprocess_peak",
    },
    "stage_timings": _stage_d_all_stages,
    "runtime_optimizations": {
        "gold_well_parallel": GOLD_WELL_PARALLEL_REPORT,
        "hjyact_deterministic_feature_reuse": HJYACT_DETERMINISTIC_REUSE_MANIFEST,
        "ridge_memory_release": RIDGE_MEMORY_RELEASE_REPORT,
        "pre_exp413_memory_release": EXP413_PRE_RELEASE_REPORT,
        "shared_bank_final_release": SHARED_BANK_FINAL_RELEASE_REPORT,
        "shared_likpf_sp45_streaming": SHARED_SP45_STREAMING_REPORT,
        "dataframe_ownership_transfer": True,
    },
    "v2_output_equivalence": _stage_d_v2_equivalence_manifest,
    "fixed_overhead_seconds": round(_stage_d_fixed_overhead_seconds, 6),
    "target_hidden_wells": _stage_d_target_wells,
    "estimation": {
        "parallel_worker_reference": _stage_d_parallel_workers,
        "parallel_lower_formula": "stage_seconds_times_200_div_4",
        "parallel_upper_formula": "stage_seconds_times_200_div_visible_wells",
        "sequential_formula": "stage_seconds_times_200_div_visible_wells",
        "fixed_overhead_policy": "add_once",
        "lower_seconds": round(_stage_d_lower_seconds, 6),
        "upper_seconds": round(_stage_d_upper_seconds, 6),
        "lower_hours": round(_stage_d_lower_seconds / 3600.0, 6),
        "upper_hours": round(_stage_d_upper_seconds / 3600.0, 6),
        "upper_limit_seconds": 32400,
        "hidden_runtime_observed": False,
        "uncertainty": "high_visible_workload_may_not_match_hidden",
    },
    "submission": {
        "generated": bool(FINAL_SUBMISSION_PATH.is_file()),
        "rows": int(len(submission)),
        "sha256": sha256_file(FINAL_SUBMISSION_PATH),
        "external_submission_performed": False,
    },
}
_stage_d_report_path = WORKING_DIR / "exp514_stage_d_visible_runtime_report.json"
_stage_d_report_path.write_text(
    json.dumps(_stage_d_report, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
metrics["stage_d_visible_runtime"] = _stage_d_report
metrics["sha256"]["stage_d_visible_runtime_report"] = sha256_file(
    _stage_d_report_path
)
metrics["sha256"]["stage_d_v2_output_equivalence"] = sha256_file(
    _stage_d_v2_equivalence_path
)
(WORKING_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
_stage_d_repro_path = WORKING_DIR / "exp514_reproducibility_manifest.json"
if _stage_d_repro_path.is_file():
    _stage_d_repro = json.loads(_stage_d_repro_path.read_text(encoding="utf-8"))
    _stage_d_repro["stage_d_visible_runtime"] = _stage_d_report
    _stage_d_repro_path.write_text(
        json.dumps(_stage_d_repro, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

print("Stage D visible runtime report:", json.dumps(_stage_d_report, sort_keys=True))
print("Stage D external submission performed: False")
display(_stage_d_report)
'''


def build_source() -> str:
    observed = sha256(BASE_SOURCE)
    if observed != EXPECTED_BASE_SOURCE_SHA256:
        raise RuntimeError(
            "base candidate source SHA drifted; refusing Stage D generation: "
            f"{observed} != {EXPECTED_BASE_SOURCE_SHA256}"
        )
    generator_sha = sha256(Path(__file__))
    source = BASE_SOURCE.read_text(encoding="utf-8")
    source = replace_once(
        source,
        f'EXP514_GENERATOR_SHA256 = "{EXPECTED_BASE_GENERATOR_SHA256}"\n',
        (
            f'EXP514_GENERATOR_SHA256 = "{EXPECTED_BASE_GENERATOR_SHA256}"\n'
            f'STAGE_D_GENERATOR_SHA256 = "{generator_sha}"\n'
            f'STAGE_D_BASE_CANDIDATE_SHA256 = "{EXPECTED_BASE_SOURCE_SHA256}"\n'
            'STAGE_D_RUNTIME_REVISION = 4\n'
            'STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256 = '
            f'"{EXPECTED_STAGE_D_V1_HJYACT_SHA256}"\n'
            'STAGE_D_V2_GOLD_BALANCED_SHA256 = '
            f'"{EXPECTED_STAGE_D_V2_GOLD_BALANCED_SHA256}"\n'
            'STAGE_D_V2_EXP413_COMPONENT_SHA256 = '
            f'"{EXPECTED_STAGE_D_V2_EXP413_COMPONENT_SHA256}"\n'
            'STAGE_D_V2_COMPONENT_READOUT_SHA256 = '
            f'"{EXPECTED_STAGE_D_V2_COMPONENT_READOUT_SHA256}"\n'
            'STAGE_D_V2_FINAL_SUBMISSION_SHA256 = '
            f'"{EXPECTED_STAGE_D_V2_FINAL_SUBMISSION_SHA256}"\n'
        ),
    )
    source = replace_once(
        source,
        PARENT_VISIBLE_HJYACT_GUARD,
        STAGE_D_CANDIDATE_VISIBLE_HJYACT_GUARD,
    )
    source = replace_once(
        source,
        "from typing import Any\n\nfrom IPython.display import display\n",
        (
            "from typing import Any\n\n"
            "import ctypes as _exp514_ctypes\n"
            "import gc as _exp514_gc\n\n"
            "from IPython.display import display\n\n"
            "STAGE_D_VISIBLE_STARTED = time.time()\n\n"
            "def _exp514_current_rss_mib():\n"
            "    status_path = Path('/proc/self/status')\n"
            "    if not status_path.is_file():\n"
            "        return None\n"
            "    for line in status_path.read_text(encoding='utf-8').splitlines():\n"
            "        if line.startswith('VmRSS:'):\n"
            "            return float(line.split()[1]) / 1024.0\n"
            "    return None\n\n\n"
            "def _exp514_release_globals(names, *, label):\n"
            "    before_mib = _exp514_current_rss_mib()\n"
            "    released = []\n"
            "    namespace = globals()\n"
            "    for name in names:\n"
            "        if name in namespace:\n"
            "            namespace.pop(name)\n"
            "            released.append(name)\n"
            "    collected = int(_exp514_gc.collect())\n"
            "    malloc_trim_called = False\n"
            "    try:\n"
            "        malloc_trim_called = bool(\n"
            "            _exp514_ctypes.CDLL('libc.so.6').malloc_trim(0)\n"
            "        )\n"
            "    except Exception:\n"
            "        malloc_trim_called = False\n"
            "    after_mib = _exp514_current_rss_mib()\n"
            "    report = {\n"
            "        'label': str(label),\n"
            "        'released_names': released,\n"
            "        'gc_collected': collected,\n"
            "        'malloc_trim_called': malloc_trim_called,\n"
            "        'rss_before_mib': before_mib,\n"
            "        'rss_after_mib': after_mib,\n"
            "    }\n"
            "    print('memory release report:', report, flush=True)\n"
            "    return report\n"
        ),
    )
    source = replace_once(
        source,
        "HJYACT_SHARED_FEATURE_RUNTIME_SECONDS = None\n",
        (
            "HJYACT_SHARED_FEATURE_RUNTIME_SECONDS = None\n"
            "HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS = None\n"
        ),
    )
    source = replace_once(
        source,
        '    print(f"submission.csv written ({len(sample)} rows) in {time.time() - t0:.0f}s")\n',
        (
            '    globals()["HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS"] = time.time() - t0\n'
            '    print(f"submission.csv written ({len(sample)} rows) in '
            '{HJYACT_LEARNED_TOTAL_RUNTIME_SECONDS:.0f}s")\n'
        ),
    )
    source = apply_memory_lifetime_patches(source)
    source = source.rstrip() + RUNTIME_REPORT_CELL
    ast.parse(source)
    return source


def main() -> None:
    source = build_source()
    OUTPUT_SOURCE.write_text(source, encoding="utf-8")
    print(f"wrote {OUTPUT_SOURCE.relative_to(ROOT)}")
    print(f"lines={len(source.splitlines())}")
    print(f"sha256={hashlib.sha256(source.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
