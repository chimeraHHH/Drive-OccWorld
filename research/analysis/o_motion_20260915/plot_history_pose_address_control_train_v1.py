#!/usr/bin/env python3
"""Plot completed descriptive summaries; no new statistics or inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'history_pose_address_control_train_analysis_v1.json'
SOURCE_SHA256 = '3f3067c189bd5121dd2e39993faae469f155c3d28ba31995388c2e0c7076f2f4'
STEM = ROOT / 'figures' / 'history_pose_address_control_train_v1'
CONTRASTS = (
    ('actual_minus_broken_delta_eD_minus_eGT', 'GT vs D', '#2166AC', 'o'),
    ('actual_minus_broken_delta_eZero_minus_eGT', 'GT vs zero', '#BA5B3A', 's'),
)
AUC_FEATURES = (
    ('score_true', 'True ZNCC score', '#2166AC'),
    ('score_broken', 'Broken score', '#8B6AA8'),
    ('lsq_speed_xy_mps', 'D speed', '#D17B27'),
)
BIN_LABELS = ('[0, 0.1)', '[0.1, 0.5)', '[0.5, 1)', '[1, 2)', '[2, 5)', '[5, 10)', '[10, ∞)')


def extract():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, 'Frozen analysis changed'
    source = json.loads(raw)
    assert source['status'] == 'COMPLETE_CPU_HISTORY_POSE_ADDRESS_CONTROL_TRAIN_ANALYSIS'
    assert (source['samples'], source['scenes']) == (512, 256)
    horizon = next(row for row in source['horizons'] if row['horizon_seconds'] == 2.)
    bins = horizon['strata']['no_radar']['all']['bins']
    full = next(row for row in bins if row['all_speeds'])['common_reference_valid']
    assert full['points'] == 253253
    speed_rows = []
    for bi, label in enumerate(BIN_LABELS):
        row = next(row for row in bins if row['bin_index'] == bi)['common_reference_valid']
        metrics = row['dimensionless_metrics']
        for key, *_ in CONTRASTS:
            assert metrics[key]['points'] == row['points']
            assert metrics[key]['weight_sum'] == row['weight_sum']
            assert metrics[key]['original_object_count'] == full['original_object_count']
        speed_rows.append(dict(bin_index=bi, interval=label, common_points=row['points'],
                               original_weight_sum=row['weight_sum'],
                               contrasts={key: metrics[key] for key, *_ in CONTRASTS}))
    assert sum(row['common_points'] for row in speed_rows) == full['points']
    auc = full['secondary_actual_past_benefit_auc']
    support = ('positive_weight', 'negative_weight', 'positive_points', 'negative_points')
    assert all({key: auc[feature][key] for key in support} ==
               {key: auc['score_true'][key] for key in support} for feature, *_ in AUC_FEATURES)
    return dict(
        schema='history-pose-address-control-train-figure-data-v1', source_path=str(SOURCE),
        source_sha256=SOURCE_SHA256, source_status=source['status'], samples=512, scenes=256,
        horizon_seconds=2., population='No radar AND CRN-uncovered; original 2 s future-valid objects, all original GT groups',
        common_population='Original camera-valid AND GT-reference-valid; identical support across metrics and AUC scores in each cell',
        common_points=full['points'], original_object_count=full['original_object_count'],
        original_point_weight=source['rules']['point_weight'],
        panel_a=dict(metric='Original-object-point-weighted conditional mean of actual-minus-broken correlation contrast',
                     units='dimensionless correlation difference',
                     definitions=source['rules']['metric_definitions'],
                     bin_edges_mps=source['rules']['speed_edges_mps'], rows=speed_rows),
        panel_b=dict(metric='Secondary weighted AUC', label=source['rules']['secondary_auc']['label'],
                     label_units='m', score_directions=source['rules']['secondary_auc']['directions'],
                     zero_benefit_points=full['secondary_actual_past_zero_benefit_points'],
                     results={key: auc[key] for key, *_ in AUC_FEATURES}),
        limitations=[
            'Descriptive statistics only; no CI, significance test or bootstrap',
            'GT reference is privileged, not a deployable signal or mathematical upper bound',
            'Only the 2 s future-valid subset is plotted; four horizon summaries are not independent',
            'Correlation contrasts are dimensionless, not physical EPE',
            'Secondary AUC predicts signed benefit at the actual past-camera time, not future occupancy IoU',
            'No claim of model, routing, motion or occupancy improvement',
        ],
    )


def render(data):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#727B84', 'axes.linewidth': .7, 'text.color': '#233142',
        'axes.labelcolor': '#233142', 'xtick.color': '#435163', 'ytick.color': '#435163',
        'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(12.2, 4.9), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=.077, right=.967, bottom=.265, top=.73,
                           width_ratios=[1.65, 1.], wspace=.62)
    a, b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    fig.text(.077, .944, 'Historical pose-address control', fontsize=17, fontweight='bold')
    fig.text(.077, .885, 'train512 / 256 scenes  |  2 s future-valid subset  |  No radar + CRN-uncovered',
             fontsize=10.5, color='#536374')
    a.set_title('a   GT-reference contrasts', loc='left', fontweight='bold', pad=19)
    rows = data['panel_a']['rows']; x = list(range(7))
    a.axhline(0, linewidth=1., color='#A6AFB8', linestyle=(0, (4, 3)), zorder=0)
    for key, label, color, marker in CONTRASTS:
        y = [row['contrasts'][key]['conditional_original_weight_mean'] for row in rows]
        a.plot(x, y, color=color, marker=marker, markersize=5.5, linewidth=1.7, label=label)
    a.set(xlim=(-.25, 6.25), ylim=(-.052, .31),
          ylabel='Mean correlation contrast\n(actual − broken; dimensionless)', xlabel='Prespecified D-speed bin (m/s)')
    a.set_xticks(x, [row['interval'] + '\n' + f"n={row['common_points']:,}" for row in rows], fontsize=8)
    a.tick_params(axis='x', pad=7)
    a.yaxis.set_major_locator(MultipleLocator(.1))
    a.grid(axis='y', color='#E8ECEF', linewidth=.7)
    a.set_axisbelow(True)
    a.legend(loc='upper left', frameon=False, fontsize=9, ncol=2, handlelength=2,
             columnspacing=1.5, borderpad=.15)

    b.set_title('b   Secondary AUC', loc='left', fontweight='bold', pad=19)
    b.axvline(.5, linewidth=1., color='#A6AFB8', linestyle=(0, (4, 3)), zorder=0)
    for yi, (key, label, color) in enumerate(AUC_FEATURES):
        value = data['panel_b']['results'][key]['auc']
        b.plot(value, yi, marker='o', color=color, markersize=7, zorder=3)
        b.annotate(f'{value:.4f}', (value, yi), xytext=(9, 0), textcoords='offset points',
                   va='center', fontsize=10, color=color)
    b.set_yticks(range(3), [label for _, label, _ in AUC_FEATURES], fontsize=9)
    b.set(xlim=(.45, .69), ylim=(2.55, -.9), xlabel='Weighted AUC\n(actual past-camera benefit)')
    b.set_xticks([.45, .50, .55, .60, .65])
    b.grid(axis='x', color='#E8ECEF', linewidth=.7)
    b.set_axisbelow(True)
    b.spines['left'].set_visible(False)
    b.tick_params(axis='y', length=0)
    b.text(0, .985, 'Same valid points: n = 253,253', transform=b.transAxes,
           fontsize=8.5, ha='left', va='top', color='#586779')
    b.text(.5, 2.48, 'chance = 0.5', fontsize=8, ha='center', va='bottom', color='#727D89')

    fig.text(.077, .105, 'Original object-point weights. Descriptive only; no CI. GT reference is privileged.',
             fontsize=9, color='#4E5C6D')
    fig.text(.077, .057, 'Only the 2 s subset is shown; the four horizon summaries are not independent.',
             fontsize=9, color='#4E5C6D')
    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix('.png'), dpi=240)
    fig.savefig(STEM.with_suffix('.pdf'), metadata={'Title': 'Historical pose-address control: descriptive diagnostic',
                                                   'Subject': 'Source SHA256 ' + SOURCE_SHA256})
    plt.close(fig)


if __name__ == '__main__':
    data = extract()
    render(data)
    target = STEM.with_suffix('.data.json')
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(source_sha256=SOURCE_SHA256, data_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                          outputs=[str(STEM.with_suffix(ext)) for ext in ('.png', '.pdf', '.data.json')]), indent=2))
