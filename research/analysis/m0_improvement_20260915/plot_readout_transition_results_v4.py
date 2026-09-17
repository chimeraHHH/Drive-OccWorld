"""Plot only the completed, frozen real readout/transition dev200 diagnostic.

CLI: --campaign DIR --development-summary OLD_SUMMARY_JSON
     --native-runs-root DIR --runs-root DIR --out NEW_DIR
     [--diagnostic-protocol JSON] [--validate-only]
No remote calls, torch/checkpoint loading, synthetic values, partial-result
rendering, training or threshold selection. Nothing is written before the
complete hash/identity/diagonal/10k scene-bootstrap checks pass.
"""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
PROTOCOL_SHA = '4baad13ab4d6fe0ddce740b6a345923668f3bba2e4a8b10e21bb84f7694b0d83'
NAMES = ('CC', 'CO', 'OC', 'OO')
HORIZONS = [0., .5, 1., 1.5, 2.]
EFFECTS = {
    'total_O_minus_C': [-1, 0, 0, 1],
    'readout_on_C_features': [-1, 1, 0, 0],
    'features_under_O_readout': [0, -1, 0, 1],
    'features_under_C_readout': [-1, 0, 1, 0],
    'readout_on_O_features': [0, 0, -1, 1],
    'interaction_on_metric_scale': [1, -1, -1, 1],
}
FOREST = ('readout_on_C_features', 'features_under_O_readout',
          'features_under_C_readout', 'readout_on_O_features',
          'total_O_minus_C', 'interaction_on_metric_scale')
FOREST_LABELS = ('Readout on C features: CO − CC', 'Features under O readout: OO − CO',
                 'Features under C readout: OC − CC', 'Readout on O features: OO − OC',
                 'Total: OO − CC', 'Interaction: OO − OC − CO + CC')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def verify(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), 'Missing complete real input; no figure: '+str(path))
        digest = sha(path)
        require(expected is None or digest == expected, 'Input SHA differs: '+str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, 'Input changed while reading')
        self.hashes[str(path)] = digest
        return path

    def read(self, path, expected=None):
        return json.loads(self.verify(path, expected).read_text())

    def rows(self, path, expected):
        return [json.loads(line) for line in self.verify(path, expected).read_text().splitlines() if line.strip()]

    def unchanged(self):
        for path, digest in self.hashes.items():
            require(sha(path) == digest, 'Input changed before rendering: '+path)


def equal_numbers(actual, expected, label):
    """Integer/string/identity fields exact; finite numerical recomputations close."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Dictionary differs: '+label)
        for key in expected:
            equal_numbers(actual[key], expected[key], label+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'List differs: '+label)
        for i, value in enumerate(expected):
            equal_numbers(actual[i], value, label+'.'+str(i))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(expected) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), 'Finite number differs: '+label)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Value differs: '+label)


def hist_array(value, shape=(5, 2, 2)):
    array = np.asarray(value)
    require(array.shape == shape and np.issubdtype(array.dtype, np.integer) and
            np.all(array >= 0) and np.all(array <= 2**53), 'Invalid integer GT-row/prediction-column counts')
    return array.astype(np.int64)


def module(name, policy, inputs):
    path = inputs.verify(PKG/name, policy['source_sha256'][name])
    spec = importlib.util.spec_from_file_location('_readout_plot_'+path.stem, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def stage(root, mode, complete_sha, policy, inputs):
    require(not (root/'failed.json').exists(), 'Failed stage cannot be plotted')
    done = inputs.read(root/'complete.json', complete_sha)
    samples, scenes = (2, 2) if mode == 'pilot' else (200, 100)
    status = 'PASS_READOUT_TRANSITION_ENGINEERING' if mode == 'pilot' else 'COMPLETE_READOUT_TRANSITION_DIAGNOSTIC'
    require(done['schema'] == 'readout-transition-complete-v1' and done['status'] == status and
            done['mode'] == mode and done['diagnostic_protocol_sha256'] == PROTOCOL_SHA and
            done['samples'] == samples and done['scenes'] == scenes and done['optimizer_updates'] == 0,
            'Wrong/incomplete diagnostic stage')
    require(set(done['files_sha256']) == {'manifest.json', 'records.jsonl', 'summary.json', 'report.md'},
            'Missing stage artifact')
    for name, digest in done['files_sha256'].items():
        inputs.verify(root/name, digest)
    summary = inputs.read(root/'summary.json', done['files_sha256']['summary.json'])
    manifest = inputs.read(root/'manifest.json', done['files_sha256']['manifest.json'])
    require(summary['schema'] == 'readout-transition-summary-v1' and manifest['schema'] == 'readout-transition-manifest-v1',
            'Wrong stage summary/manifest schema')
    for item in (summary, manifest):
        require(item['mode'] == mode and item['diagnostic_protocol_sha256'] == PROTOCOL_SHA and item['optimizer_updates'] == 0,
                'Stage scope/protocol changed')
    require(summary['samples'] == samples and summary['scenes'] == scenes and
            summary['all_engineering_gates_passed'] is True and summary['model_and_input_state_unchanged'] is True,
            'Engineering gates did not pass')
    require(summary['sources']['source_sha256'] == manifest['source_sha256'] == policy['source_sha256'] and
            summary['sources']['script_sha256'] == policy['source_sha256']['readout_transition_diagnostic_v4.py'] and
            summary['sources']['diagnostic_protocol_sha256'] == PROTOCOL_SHA and
            summary['sources']['cache_index_sha256'] == manifest['cache_index_sha256'] == policy['cache_index_sha256'],
            'Producer/source/cache changed')
    require(summary['sources']['development_summary'] == manifest['development_summary'] == policy['development_summary'] and
            summary['selected_identities'] == manifest['selected_identities'] and
            summary['loaded_models'] == manifest['loaded_models'], 'Stage source/selection receipt differs')
    require(manifest['parent_protocol_sha256'] == summary['sources']['parent_protocol_sha256'] == policy['parent_protocol_sha256'] and
            manifest['objective_protocol_sha256'] == summary['sources']['objective_protocol_sha256'] == policy['objective_protocol_sha256'],
            'Parent source differs')
    require(set(summary['loaded_models']) == {'C', 'O'}, 'Only fixed C/O weights permitted')
    for name, loaded in summary['loaded_models'].items():
        bound = policy['model_sources'][name]
        require(loaded['checkpoint_sha256'] == bound['files_sha256']['latest.pth'] and
                loaded['final_head_state_sha256'] == bound['final_head_state_sha256'] and
                loaded['loss_adapter_installed'] is False and loaded['optimizer_loaded'] is False and
                loaded['optimizer_constructed'] is False and is_sha(loaded['full_model_state_sha256']) and
                is_sha(loaded['complete_readout_state_sha256']), 'Actual loaded-model receipt differs')
    semantics = summary['semantics']
    for key, expected in dict(posthoc=True, historical_validation_exposure=True, training_seeds=1,
            full_validation=False, new_candidate_selection=False, unique_causal_attribution=False,
            latent_reparameterization_and_coadaptation_caveat=True, features_or_logits_persisted=False).items():
        require(semantics[key] == expected, 'Scientific interpretation differs: '+key)
    records = inputs.rows(root/'records.jsonl', done['files_sha256']['records.jsonl'])
    require(len(records) == samples, 'Incomplete stage row count')
    return done, summary, manifest, records


def validate_rows(rows, identities, ordinals, diagonal):
    require(len(rows) == len(identities) == len(ordinals), 'Stage row lengths differ')
    for row, identity, ordinal in zip(rows, identities, ordinals):
        require(row['schema'] == 'readout-transition-sample-v1' and row['ordinal'] == ordinal and
                {key: row[key] for key in identity} == identity and row['horizon_seconds'] == HORIZONS,
                'Wrong diagnostic identity/order/horizons')
        counts = hist_array(identity['target_counts_0_1_255_by_frame'], (7, 3))
        require(np.all(counts.sum(1) == 512*512*40), 'GT counts do not exhaust each original grid')
        require(row['common_replay_valid_frames'] == [] and all(row[key] is True for key in
                ('original_replay_clones_boundary', 'input_and_target_unchanged', 'features_unchanged',
                 't0_features_C_equal_O_bitwise', 't0_all_layers_OC_equal_CC_bitwise', 't0_all_layers_CO_equal_OO_bitwise')),
                'Per-anchor replay/t0 gates absent')
        require(is_sha(row['inputs_full_typed_sha256']) and is_sha(row['target_tensor_sha256']) and
                set(row['hist_by_combination']) == set(row['combinations']) == set(NAMES), 'Missing source/hash/combination')
        for name in NAMES:
            matrix = hist_array(row['hist_by_combination'][name]); receipt = row['combinations'][name]
            require(np.array_equal(matrix.sum(2), counts[2:, :2]), 'Combination changes GT valid counts')
            require(receipt['feature_source'] == name[0] and receipt['readout_source'] == name[1] and
                    is_sha(receipt['logits_sha256']) and is_sha(receipt['feature_sha256']), 'Wrong feature/readout source')
            if name[0] == name[1]:
                require(receipt['diagonal_same_call_native_all_5h_3layer_logits_bitwise_equal'] is True and
                        receipt['diagonal_historical_all_5h_confusion_equal'] is True and
                        receipt['same_call_native_logits_sha256'] == receipt['logits_sha256'] and
                        np.array_equal(matrix, diagonal[name[0]][ordinal]), 'Historical diagonal changed')
            else:
                require(receipt['diagonal_same_call_native_all_5h_3layer_logits_bitwise_equal'] is None and
                        receipt['diagonal_historical_all_5h_confusion_equal'] is None, 'Cross must not claim native diagonal parity')
        require(row['combinations']['CC']['feature_sha256'] == row['combinations']['CO']['feature_sha256'] and
                row['combinations']['OC']['feature_sha256'] == row['combinations']['OO']['feature_sha256'], 'Features changed across readouts')
        require(row['hist_by_combination']['CC'][0] == row['hist_by_combination']['OC'][0] and
                row['hist_by_combination']['CO'][0] == row['hist_by_combination']['OO'][0], 't0 confusions violate exact gate')


def load_inputs(args):
    inputs = Inputs(); root = Path(args.campaign).resolve()
    require(not (root/'failed.json').exists(), 'Failed/incomplete campaign cannot be plotted')
    campaign = inputs.read(root/'complete.json')
    require(campaign['status'] == 'COMPLETE_READOUT_TRANSITION_CAMPAIGN' and
            campaign['diagnostic_protocol_sha256'] == PROTOCOL_SHA and campaign['optimizer_steps'] == 0 and
            campaign['automatic_retry'] is False and campaign['model_selection'] is False,
            'Require complete real readout campaign, not old O/full/pilot results')
    policy = inputs.read(args.diagnostic_protocol, PROTOCOL_SHA)
    require(policy['schema'] == 'readout-transition-diagnostic-protocol-v1' and policy['status'] == 'FROZEN' and
            policy['combinations'] == list(NAMES), 'Wrong fixed diagnostic protocol')
    for name, digest in policy['source_sha256'].items():
        require(Path(name).name == name, 'Unsafe source path'); inputs.verify(PKG/name, digest)
    manifest = inputs.read(root/'manifest.json', campaign['manifest_sha256'])
    require(manifest['schema'] == 'readout-transition-campaign-v1' and manifest['diagnostic_protocol_sha256'] == PROTOCOL_SHA and
            manifest['wrapper_sha256'] == policy['source_sha256']['run_readout_transition_campaign_v4.py'] and
            manifest['dependency_gate_sha256'] == policy['source_sha256']['objective_full_dependency.py'] and
            manifest['resources'] == policy['resources'], 'Campaign producer/resource binding differs')
    dependency = inputs.read(root/'dependency_complete.json', campaign['dependency_receipt_sha256'])
    require(dependency['runner_status'] == 'EXITED_ZERO' and dependency['runner_returncode'] == 0 and
            dependency['samples'] == 5119 and dependency['scenes'] == 150 and
            dependency['selected_from_partial_performance'] is False and dependency['performance_gate_applied'] is False and
            all(is_sha(dependency[key]) for key in ('campaign_complete_sha256', 'merge_complete_sha256', 'summary_sha256', 'job_state_sha256')),
            'Full dependency did not finish under its bound completion gate')
    require(Path(dependency['campaign']) == Path(manifest['full_campaign']) and
            Path(dependency['job_binding']['job']) == Path(manifest['full_job']), 'Dependency path differs')
    old_path = Path(args.development_summary); old_done = inputs.read(old_path.with_name('complete.json'), policy['development_summary']['complete_sha256'])
    require(old_done['status'] == 'COMPLETE' and old_done['summary_sha256'] == policy['development_summary']['summary_sha256'],
            'Old certified development result incomplete')
    for name, digest in old_done['files_sha256'].items():
        require(Path(name).name == name, 'Unsafe old artifact path'); inputs.verify(old_path.parent/name, digest)
    old = inputs.read(old_path, policy['development_summary']['summary_sha256'])
    require(old['schema'] == 'm0-objective-supervision-aggregate-v1' and old['samples'] == 200 and old['scenes'] == 100 and
            old['audit']['status'] == 'PASS' and old['audit']['actual_three_final_checkpoints_rehashed_and_payload_checked_on_CPU'] is True,
            'Old C/O payload/GT audit not certified')
    identities = old['ordered_identities_and_GT']; scenes = Counter(row['scene_token'] for row in identities)
    require(len(identities) == len({row['sample_token'] for row in identities}) == 200 and
            len(scenes) == 100 and set(scenes.values()) == {2} and all(row['split'] == 'development' for row in identities),
            'Expected all 200 development anchors, two per scene')
    diagonal = {}
    for name, oldname, directory in [('C', 'native1', Path(args.native_runs_root)/'native1'), ('O', 'O', Path(args.runs_root)/'O')]:
        source = old['source_receipts'][oldname]; files = source['reuse_audit']['files_sha256'] if name == 'C' else source['files_sha256']
        require(policy['model_sources'][name] == dict(files_sha256=files, final_head_state_sha256=source['checkpoint']['final_head_state_sha256']),
                'Policy differs from original actual-checkpoint audit')
        for filename in ('manifest.json', 'training_complete.json', 'complete.json'):
            inputs.verify(directory/filename, files[filename])
        prior = inputs.rows(directory/'development_records.jsonl', files['development_records.jsonl'])
        require(len(prior) == 200, 'Incomplete original diagonal rows')
        for row, identity in zip(prior, identities):
            require(row['sample_token'] == identity['sample_token'] and row['scene_token'] == identity['scene_token'] and
                    row['horizon_seconds'] == HORIZONS, 'Original diagonal order differs')
        diagonal[name] = [hist_array(row['hist_by_horizon']) for row in prior]
    pd, ps, pm, pr = stage(root/'pilot', 'pilot', campaign['pilot_complete_sha256'], policy, inputs)
    dd, summary, dm, rows = stage(root/'development', 'development', campaign['development_complete_sha256'], policy, inputs)
    producer = module('readout_transition_diagnostic_v4.py', policy, inputs)
    aggregate = module('objective_supervision_aggregate.py', policy, inputs)
    metric = module('aggregate_memory.py', policy, inputs)
    pilot_ordinals = producer.first_two_scenes(identities)
    require(pm['selected_ordinals'] == pilot_ordinals and ps['selected_identities'] == [identities[i] for i in pilot_ordinals] and
            dm['selected_ordinals'] == list(range(200)) and summary['selected_identities'] == identities and
            summary['loaded_models'] == ps['loaded_models'], 'Pilot/development identity or model changed')
    validate_rows(pr, ps['selected_identities'], pilot_ordinals, diagonal); validate_rows(rows, identities, list(range(200)), diagonal)
    auth = inputs.read(root/'development_authorization.json', campaign['development_authorization_sha256'])
    r = ps['resources']; init, sample = r['pre_sample_initialization_seconds'], r['max_sample_seconds']
    require(math.isfinite(init) and init >= 0 and math.isfinite(sample) and sample > 0, 'Invalid pilot timing')
    cap = max(300, math.ceil(1.75*(init+200*sample)+60))
    require(auth['schema'] == 'readout-transition-development-authorization-v1' and auth['status'] == 'FROZEN' and
            auth['diagnostic_protocol_sha256'] == PROTOCOL_SHA and auth['pilot_complete_sha256'] == campaign['pilot_complete_sha256'] and
            auth['pilot_summary_sha256'] == pd['files_sha256']['summary.json'] and auth['resources'] == dm['resources'] == dict(max_seconds=cap, max_allocated_gib=32) and
            cap <= 3000 and pm['resources'] == dict(max_seconds=300, max_allocated_gib=32), 'Pilot-derived budget/authorization differs')
    require(summary['phase_gate'] == dm['phase_gate'] == dict(authorization_sha256=campaign['development_authorization_sha256'],
            pilot_complete_sha256=campaign['pilot_complete_sha256'], pilot_summary_sha256=pd['files_sha256']['summary.json'], pilot_seconds=pd['seconds']),
            'Development did not consume the same pilot PASS')
    require(producer.EFFECTS == EFFECTS and producer.NAMES == NAMES, 'Frozen combination/effect math changed')
    models, effects, bootstrap = producer.summarize(rows, aggregate, metric)
    equal_numbers(summary['models'], models, 'models'); equal_numbers(summary['effects'], effects, 'effects')
    equal_numbers(summary['bootstrap'], bootstrap, 'bootstrap')
    require(bootstrap['replicates'] == 10000 and bootstrap['seed'] == 11 and bootstrap['scene_count'] == 100 and
            bootstrap['paired'] is True and bootstrap['all_anchors_within_scene_kept_together'] is True,
            'Wrong scene uncertainty calculation')
    for name, oldname in [('CC', 'native1'), ('OO', 'O')]:
        require(models[name]['hist_by_horizon'] == [h['confusion_GT_rows_prediction_columns'] for h in old['models'][oldname]['horizons']],
                'Original pooled diagonal changed')
        equal_numbers(models[name]['metrics'], old['models'][oldname]['metrics'], 'historical '+name)
    values = {name: effects[name]['metrics']['future_macro_gmo']['delta_ratio'] for name in EFFECTS}
    require(math.isclose(values['readout_on_C_features']+values['features_under_O_readout'], values['total_O_minus_C'], abs_tol=1e-12) and
            math.isclose(values['features_under_C_readout']+values['readout_on_O_features'], values['total_O_minus_C'], abs_tol=1e-12),
            'The two conditional paths do not add to the same total')
    inputs.unchanged()
    return inputs, summary


def numbers(out, summary):
    with (out/'horizon_numbers.csv').open('x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['combination', 'feature_source', 'readout_source', 'horizon_seconds',
            'GMO_ratio', 'GMO_percent', 'binary_mIoU_ratio', 'TN', 'FP', 'FN', 'TP'])
        for name in NAMES:
            model = summary['models'][name]
            for h, seconds in enumerate(HORIZONS):
                matrix = model['hist_by_horizon'][h]
                writer.writerow([name, name[0], name[1], seconds, model['gmo_iou_by_horizon'][h],
                    100*model['gmo_iou_by_horizon'][h], model['binary_miou_by_horizon'][h], *matrix[0], *matrix[1]])
    with (out/'effect_numbers.csv').open('x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['effect', 'coefficients_CC_CO_OC_OO', 'metric', 'horizon_seconds',
            'delta_ratio', 'delta_pp', 'CI95_lower_pp', 'CI95_upper_pp', 'plotted_in_forest'])
        for name, effect in summary['effects'].items():
            for metric, value in effect['metrics'].items():
                is_horizon = metric == 'gmo_iou_by_horizon'
                for h in range(5) if is_horizon else [None]:
                    writer.writerow([name, json.dumps(effect['coefficients'], sort_keys=True), metric,
                        HORIZONS[h] if h is not None else '',
                        value['delta_ratio'][h] if is_horizon else value['delta_ratio'],
                        value['delta_pp'][h] if is_horizon else value['delta_pp'],
                        value['ci95_pp'][0][h] if is_horizon else value['ci95_pp'][0],
                        value['ci95_pp'][1][h] if is_horizon else value['ci95_pp'][1], metric == 'future_macro_gmo'])


def render(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#75818B',
        'text.color': '#24313B', 'axes.labelcolor': '#24313B', 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(15.4, 7.8), facecolor='white')
    grid = fig.add_gridspec(1, 2, width_ratios=[1., 1.05], left=.065, right=.97, bottom=.28, top=.78, wspace=.78)
    left, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    colors = {'C': '#3775AA', 'O': '#137F77'}; markers = ('o', 's', '^', 'D')
    for name, marker in zip(NAMES, markers):
        left.plot(HORIZONS, 100*np.asarray(summary['models'][name]['gmo_iou_by_horizon']),
            color=colors[name[1]], linestyle='-' if name[0] == 'C' else '--', marker=marker,
            markersize=5.5, linewidth=1.8, markerfacecolor='white',
            label=name+': '+name[0]+' features / '+name[1]+' readout')
    left.set(title='(a) Four fixed recombinations', xlabel='Forecast horizon (s)', ylabel='GMO IoU (%)', xticks=HORIZONS)
    left.grid(axis='y', color='#DFE5E8', linewidth=.7); left.legend(loc='upper right', fontsize=8.6, frameon=False)
    left.text(0, -.22, 't = 0 is shown; only the four future horizons enter the forest.', transform=left.transAxes, fontsize=8.7)
    left.text(0, -.29, 'Line: feature source C (solid), O (dashed). Color: readout.', transform=left.transAxes, fontsize=8.7)
    forest_colors = ['#3775AA']*2+['#137F77']*2+['#24313B', '#7655A3']
    all_limits = [0.]
    for i, name in enumerate(FOREST):
        value = summary['effects'][name]['metrics']['future_macro_gmo']; point = value['delta_pp']; lo, hi = value['ci95_pp']
        right.plot([lo, hi], [i, i], color=forest_colors[i], linewidth=2)
        right.plot(point, i, marker='D' if i >= 4 else 'o', color=forest_colors[i], markersize=5.5)
        all_limits.extend([point, lo, hi])
    span = max(max(all_limits)-min(all_limits), .1)
    right.set_xlim(min(all_limits)-.1*span, max(all_limits)+.1*span)
    right.axvline(0, color='#89949D', linewidth=.8); right.axhline(1.5, color='#DEE5E8', linewidth=.7)
    right.axhline(3.5, color='#DEE5E8', linewidth=.7)
    right.set(yticks=range(6), yticklabels=FOREST_LABELS, xlabel='Difference in future macro GMO (pp)',
              title='(b) Conditional effects and metric interaction', ylim=(5.65, -.65))
    right.tick_params(axis='y', length=0, labelsize=9)
    right.text(0, -.22, 'Path A: CC → CO → OO      Path B: CC → OC → OO', transform=right.transAxes, fontsize=8.7)
    right.text(0, -.29, 'Both path sums equal OO − CC; intervals are not additive.', transform=right.transAxes, fontsize=8.7)
    fig.text(.065, .935, 'Feature / readout recombination after fixed C and O training', fontsize=17, weight='bold')
    fig.text(.065, .885, 'C = original-objective continuation   ·   O = CE[1,5] + Lovász continuation   ·   same final checkpoints', fontsize=10.5)
    fig.text(.065, .115, 'Post-hoc diagnostic, not a causal allocation: cross-model latent coordinates and readouts may co-adapt.', fontsize=10, weight='bold')
    fig.text(.065, .075, '200 exposed development anchors / 100 scenes; one training seed (11). Bars: unadjusted 95% paired-scene CI, 10,000 draws.', fontsize=9)
    fig.text(.065, .04, 'GMO denotes movable semantic categories, not measured motion. Fixed argmax; no threshold fitting, new candidate promotion or full-validation claim.', fontsize=9)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out/('readout_transition_results.'+extension), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'development-summary', 'native-runs-root', 'runs-root', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--diagnostic-protocol', default=str(PKG/'readout_transition_diagnostic_protocol_v4.json'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv); out = Path(args.out).resolve()
    require(not out.exists(), 'Output must be new; source results are never overwritten')
    inputs, summary = load_inputs(args)
    if args.validate_only:
        print(json.dumps(dict(status='VALIDATED_REAL_COMPLETED_INPUTS_NO_RENDER', samples=200, scenes=100,
                             bootstrap_recomputed=True, output_written=False))); return
    out.mkdir(parents=True, exist_ok=False)
    numbers(out, summary); render(out, summary)
    caption = ('Post-hoc feature/readout recombination on the same 200 development anchors in 100 scenes. '
        'The first letter gives the future-feature source; the second gives the complete terminal readout source. '
        'CC and OO reproduce the original C and O five-horizon full-GT confusion matrices exactly. '
        '(a) All five horizons, including current time, are shown without smoothing or synthetic values. '
        '(b) Four conditional effects, total OO−CC, and OO−OC−CO+CC are calculated on future macro GMO: '
        'pool full-resolution confusion across anchors separately at each future horizon, compute class-1 IoU, '
        'then average the four future IoUs. Path A is CC→CO→OO; path B is CC→OC→OO. Their point estimates '
        'sum to the same total, while percentile interval endpoints cannot be added. '
        'Bars are unadjusted two-sided 95% percentile intervals from 10,000 shared paired-scene draws, seed11; '
        'both anchors from a scene stay together. These intervals condition on fixed weights and exclude '
        'training-seed and historical-selection uncertainty. Cross-readout incompatibility can reflect latent '
        'reparameterization or co-adaptation: no unique causal decomposition or improved dynamics is established. '
        'One training seed, exposed development data, unchanged argmax, no threshold fitting or new candidate '
        'promotion. GMO refers to movable semantic categories, not actual motion. This is not full-validation evidence.\n')
    (out/'caption.txt').write_text(caption)
    filenames = ['readout_transition_results.png', 'readout_transition_results.pdf', 'readout_transition_results.svg',
                 'horizon_numbers.csv', 'effect_numbers.csv', 'caption.txt']
    receipt = dict(schema='readout-transition-results-plot-v4', status='RENDERED_PENDING_VISUAL_QA',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), script_sha256=sha(__file__),
        diagnostic_protocol_sha256=PROTOCOL_SHA, input_files_sha256=inputs.hashes,
        engineering_revision=4, previous_plot_script_sha256='81f6e44ac88adb50d56862043a5754c2734b1ac036c6b305e61873b2311005e0',
        combinations=list(NAMES), effects=list(EFFECTS), horizon_seconds=HORIZONS, samples=200, scenes=100,
        metrics_recomputed=True, bootstrap_recomputed=True, bootstrap=summary['bootstrap'],
        original_diagonal_200_by_5_confusions_exact=True, checkpoints_reread=False, raw_GT_or_logits_reread=False,
        missing_data_substitution=False, source_results_modified=False, posthoc=True, unique_causal_attribution=False,
        full_validation=False, training_seeds=[11], files_sha256={name: sha(out/name) for name in filenames})
    (out/'render_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'], out=str(out))))


if __name__ == '__main__':
    main()
