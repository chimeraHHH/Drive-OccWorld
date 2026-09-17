"""Plot fixed training-internal motion-evidence AUCs; no scoring or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

SOURCE_SHA = '38bb878bffc98c67715adb84f4a1a2983e5425c1e2a07d0fea1509576117b518'
STEM = 'motion_evidence_train_v1'
TITLE = 'Global AUC does not establish motion reliability'
FEATURES = (
    ('lsq_speed_xy_mps', 'LSQ speed  ↑', '#326C96'),
    ('min_horizon_speed_xy_mps', 'Minimum horizon speed  ↑', '#326C96'),
    ('max_horizon_speed_xy_mps', 'Maximum horizon speed  ↑', '#326C96'),
    ('non_cv_velocity_rms_xy_mps', 'Temporal residual RMS  ↓', '#B66A42'),
    ('relative_non_cv_velocity_rms_xy', 'Relative temporal residual  ↓', '#16828A'),
    ('local_3x3_velocity_rms_xy_mps', 'Local 3×3 velocity RMS  ↓', '#917094'),
    ('local_3x3_predicted_coverage_fraction', 'Local 3×3 coverage  ↑', '#7B8791'),
)
BIN_FEATURES = (FEATURES[0], FEATURES[3], FEATURES[4], FEATURES[5])
BIN_LABELS = ('[0,0.1)', '[0.1,0.5)', '[0.5,1)', '[1,2)', '[2,5)', '[5,10)', '[10,∞)')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact_feature(row):
    return {k: v for k, v in row.items() if k != 'per_scene'}


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=base/'motion_evidence_train_analysis_v1.json')
    p.add_argument('--out-dir', type=Path, default=base/'figures')
    args = p.parse_args()
    require(sha(args.source) == SOURCE_SHA, 'Fixed analysis SHA differs')
    data = json.loads(args.source.read_text())
    require(data['status'] == 'COMPLETE_CPU_FIXED_TRAIN_EVIDENCE_ANALYSIS'
            and data['samples'] == 512 and data['scenes'] == 256, 'Wrong analysis population/status')
    evidence = [e for e in data['evidence'] if e['horizon_seconds'] == 2.]
    require(len(evidence) == 1, 'Require exactly one fixed 2 s analysis')
    evidence = evidence[0]
    global_rows = {r['feature']: r for r in evidence['features']}
    require(set(global_rows) == {f[0] for f in FEATURES}, 'Seven-feature set differs')
    bins = evidence['speed_bins']
    edges = data['rules']['speed_control']['bin_edges_mps']
    require(edges == [0., .1, .5, 1., 2., 5., 10., None]
            and len(bins) == 7 and [b['bin_index'] for b in bins] == list(range(7)), 'Fixed bins differ')
    require(not data['rules']['auc']['direction_selection']
            and not data['rules']['inference']['new_gate']
            and not data['rules']['inference']['bootstrap'], 'Interpretation differs')
    for key, _, _ in FEATURES:
        require(global_rows[key]['direction'] == data['rules']['auc']['directions'][key], 'Score direction differs')
    matrix = np.array([[next(f['auc'] for f in b['features'] if f['feature'] == key)
                        for b in bins] for key, _, _ in BIN_FEATURES], dtype=float)
    global_auc = np.array([global_rows[f[0]]['auc'] for f in FEATURES], dtype=float)
    require(np.isfinite(matrix).all() and np.isfinite(global_auc).all()
            and (matrix >= 0).all() and (matrix <= 1).all(), 'Undefined/out-of-range selected AUC')
    support = [b['by_group']['all'] for b in bins]
    require(sum(s['points'] for s in support) == evidence['support']['points'], 'Bins omit points')
    weights = np.array([s['weight_sum'] for s in support])
    shares = 100*weights/evidence['support']['weight_sum']

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': '#1D2C39', 'axes.labelcolor': '#1D2C39',
        'xtick.color': '#536270', 'ytick.color': '#344958',
        'axes.edgecolor': '#ABB7C0', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(15.2, 7.6))
    a = fig.add_axes([.215, .34, .245, .445])
    b = fig.add_axes([.625, .445, .335, .31])
    fig.text(.037, .954, TITLE, fontsize=19, weight='bold')
    fig.text(.037, .902,
        'Training-internal diagnosis · train512 / 256 D-seen scenes · 2 s · CRN-uncovered source points',
        fontsize=11.5, color='#536270')
    fig.text(.037, .834, 'A   All seven fixed feature scores', fontsize=12, weight='bold')
    fig.text(.526, .834, 'B   Fixed speed bins: consistency versus speed control', fontsize=12, weight='bold')
    fig.text(.625, .791, 'Bins of D’s LSQ predicted speed (m/s)', fontsize=10, color='#536270')

    a.axvline(.5, color='#8F9DA8', ls=(0, (3, 3)), lw=1)
    for y, ((key, label, color), value) in enumerate(zip(FEATURES, global_auc)):
        a.plot([.5, value], [y, y], color=color, lw=2, alpha=.65)
        a.scatter(value, y, s=53, color=color, zorder=3, edgecolor='white', linewidth=.6)
        offset = -.035 if value < .5 else .035
        a.text(value+offset, y, f'{value:.3f}', va='center',
               ha='right' if value < .5 else 'left', fontsize=10.5, color=color, weight='bold')
    a.set_yticks(np.arange(7), [f[1] for f in FEATURES])
    a.tick_params(axis='y', length=0, pad=10)
    a.set_ylim(6.6, -.65); a.set_xlim(0, 1)
    a.set_xticks(np.linspace(0, 1, 6)); a.spines['left'].set_visible(False)
    a.grid(axis='x', color='#E7ECEF', lw=.65)
    a.set_xlabel('Global weighted AUC', labelpad=9)
    a.text(0, -.22, '↑ rank high / ↓ rank low; score directions fixed.',
           transform=a.transAxes, fontsize=9.4, color='#62717E')
    a.text(0, -.29, 'No confidence intervals were estimated.',
           transform=a.transAxes, fontsize=9.4, color='#62717E')

    cmap = LinearSegmentedColormap.from_list('fixed_auc', ['#B76438', '#FBFAF5', '#1E7D91'])
    heat = b.imshow(matrix, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vmin=0, vcenter=.5, vmax=1), interpolation='nearest')
    short_labels = ['LSQ speed ↑', 'Temporal RMS ↓', 'Relative residual ↓', 'Local 3×3 RMS ↓']
    b.set_yticks(np.arange(4), short_labels)
    b.set_xticks(np.arange(7), BIN_LABELS)
    b.xaxis.tick_top(); b.tick_params(axis='x', length=0, pad=8, labelsize=9)
    b.tick_params(axis='y', length=0, pad=9, labelsize=9.8)
    for spine in b.spines.values(): spine.set_visible(False)
    b.set_xticks(np.arange(-.5, 7, 1), minor=True)
    b.set_yticks(np.arange(-.5, 4, 1), minor=True)
    b.grid(which='minor', color='white', lw=2); b.tick_params(which='minor', length=0)
    for i in range(4):
        for j in range(7):
            value = matrix[i, j]
            b.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=10,
                   color='white' if value < .23 or value > .8 else '#263D48')
    # Support table is aligned to all seven bins, including the sparse last bin.
    for label, row_y, values in [
        ('Weight share', .391, [f'{v:.2f}%' for v in shares]),
        ('Positive points', .358, [f"{s['positive_points']:,}" for s in support]),
        ('Negative points', .325, [f"{s['negative_points']:,}" for s in support]),
    ]:
        fig.text(.615, row_y, label, ha='right', va='center', fontsize=9.2, color='#62717E')
        for j, value in enumerate(values):
            fig.text(.625+.335*(j+.5)/7, row_y, value, ha='center', va='center', fontsize=9)
    cax = fig.add_axes([.7, .257, .205, .015])
    cb = fig.colorbar(heat, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, pad=3, labelsize=9)
    cb.set_label('Within-bin weighted AUC · 0.5 = no ranking', fontsize=9, labelpad=4)

    fig.text(.037, .144,
        'Benefit = XY EPE(zero) − XY EPE(D); AUC ranks benefit > 0 versus benefit < 0. Zero-benefit points are excluded only from AUC (here: 0).',
        fontsize=9.4, color='#62717E')
    fig.text(.037, .11,
        'Full-object weights: each point has weight 1 / ALL valid points in its original anchor–object–horizon; no within-bin object renormalization.',
        fontsize=9.4, color='#62717E')
    fig.text(.037, .076,
        '2 s support: 353,431 uncovered source points from 8,565 objects; full original denominator: 10,332 objects. All seven fixed bins are shown.',
        fontsize=9.4, color='#62717E')
    fig.text(.037, .042,
        'D saw every training scene. Four forecast heads are not independent observations. No gate fitted, no held-out generalization or occupancy result.',
        fontsize=9.4, color='#62717E')
    fig.canvas.draw()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir/(STEM+'.png'); pdf = args.out_dir/(STEM+'.pdf')
    fig.savefig(png, dpi=200, facecolor='white', metadata={'Title': TITLE, 'Source SHA256': SOURCE_SHA})
    fig.savefig(pdf, facecolor='white', metadata={'Title': TITLE,
        'Subject': 'D-seen train512, training-internal, 2 s, uncovered-source, full-object weights; SHA256 '+SOURCE_SHA,
        'Creator': Path(__file__).name})
    plt.close(fig)
    receipt = dict(schema='motion-evidence-train-figure-v1', title=TITLE,
        source_analysis=str(args.source.resolve()), source_analysis_sha256=SOURCE_SHA,
        plot_source_sha256=sha(__file__), source_provenance=data['sources'], source_rules=data['rules'],
        source_verification=data['verification'], samples=512, scenes=256, horizon_seconds=2.,
        population=evidence['population'], support=evidence['support'],
        global_features=[compact_feature(global_rows[f[0]]) for f in FEATURES],
        within_bin_selected_feature_order=[f[0] for f in BIN_FEATURES],
        within_bin_auc_matrix=matrix.tolist(),
        all_speed_bins=[dict(bin_index=b['bin_index'],left_inclusive_mps=b['left_inclusive_mps'],
            right_exclusive_mps=b['right_exclusive_mps'],right_unbounded=b['right_unbounded'],
            all_group_support=b['by_group']['all'],features=b['features'],weight_share_percent=float(shares[i]))
            for i,b in enumerate(bins)],
        bins_omitted=[], score_direction_reselected=False, data_reestimated=False,
        fitting=False, new_gate=False, held_out_evaluation=False, model_inference=False,
        confidence_intervals_drawn=False, scene_IQR_drawn=False,
        software=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
        files_sha256={png.name:sha(png),pdf.name:sha(pdf)},
        visual_review=dict(status='PENDING_RENDER_INSPECTION'))
    output=args.out_dir/(STEM+'.json')
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(png=str(png.resolve()),pdf=str(pdf.resolve()),receipt=str(output.resolve()),
        plot_source_sha256=sha(__file__)),indent=2))


if __name__ == '__main__':
    main()
