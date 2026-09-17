"""CPU/NumPy planar physical phantom of the inspected frame contract.

Executes the exact loader's metric-coordinate transformation AST (before
voxelization), and the exact dataset row-transform expressions. Quaternion
objects are replaced only by explicit orthogonal 3x3 matrices with inverse.
Attention evidence executes the native future-to-history matrix multiplication
and point multiplication expressions, with torch.matmul replaced by np.matmul.
This is geometry evidence, not a checkpoint forward, CUDA/kernel test, or
measured occupancy benchmark. No learned offsets/features are simulated.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types
import numpy as np


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin'
OUT=Path(__file__).with_name('coordinate_contract_proof.json')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def statements(path,lo,hi):
    tree=ast.parse(path.read_text())
    chosen=[copy.deepcopy(node) for node in ast.walk(tree)
            if isinstance(node,ast.Assign) and lo<=node.lineno and node.end_lineno<=hi]
    chosen.sort(key=lambda n:n.lineno)
    assert chosen
    module=ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[]))
    return compile(module,str(path),'exec'),ast.unparse(module)


class Rotation:
    def __init__(self,R):self.rotation_matrix=np.asarray(R)
    @property
    def inverse(self):return Rotation(self.rotation_matrix.T)


def transform(angle,translation):
    c,s=np.cos(angle),np.sin(angle)
    out=np.eye(4);out[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];out[:3,3]=translation
    return out


def source_label(point_lidar,ego_poses,lidar_poses,code):
    # Matches Template.get_lidar_pose/get_ego2lidar_pose exactly:
    # negative unrotated translation + inverse quaternion.
    results=dict(egopose_list=[[-T[:3,3],Rotation(T[:3,:3]).inverse] for T in ego_poses],
                 ego2lidar_list=[[-T[:3,3],Rotation(T[:3,:3]).inverse] for T in lidar_poses])
    ns=dict(np=np,results=results,self=types.SimpleNamespace(time_history_field=0),
            count=1,pcd_np_cor=np.asarray(point_lidar).reshape(1,3))
    exec(code,ns)
    return ns['pcd_np_cor'][0]


def main():
    files=dict(loader=BASE/'datasets/pipelines/loading_occupancy.py',
               template=BASE/'datasets/nuscenes_world_dataset_template.py',
               dataset=BASE/'datasets/nuscenes_world_dataset_v1.py',
               detector=BASE/'bevformer/detectors/drive_occworld.py',
               head=BASE/'bevformer/dense_heads/world_head_v1.py',
               headbase=BASE/'bevformer/dense_heads/world_head_base.py',
               decoder=BASE/'bevformer/modules/world_decoder.py',
               transformer=BASE/'bevformer/modules/world_transformer.py',
               normalization=BASE/'bevformer/modules/conditionalnorm.py',
               grid=BASE/'bevformer/utils/e2e_predictor_utils.py',
               encoder=BASE/'bevformer/modules/encoder.py',
               config=ROOT/'analysis/sota_p2_20260911/configs/S0.py')
    assert digest(files['loader'])=='ac9190193630eca9f3118136e2d0b46bf528217f50244e37e6fc1cbff4841d1e'
    assert digest(files['detector'])=='67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
    assert digest(files['config'])=='c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303'
    loader,label_source=statements(files['loader'],68,89)
    matrices,matrix_source=statements(files['dataset'],76,87)
    # Selecting individual original assignments avoids any synthetic torch API.
    align,align_source=statements(files['detector'],345,345)
    point,point_source=statements(files['detector'],357,357)
    cases=[]
    for name,angle,translation in [('identity',0.,[0.,0.,0.]),
        ('ego_translation_plus_x_2m',0.,[2.,0.,0.]),
        ('ego_yaw_plus_90deg',np.pi/2,[0.,0.,0.]),
        ('ego_translation_and_yaw',np.pi/2,[2.,0.,0.])]:
        present=transform(0.,[0.,0.,0.]);future=transform(angle,translation)
        lidar0=lidar1=np.eye(4)
        # A fixed global point, e.g. a stationary parked car (GMO need not move).
        global_point=np.array([10.1,4.3,.1,1.])
        x0=(np.linalg.inv(present@lidar0)@global_point)[:3]
        x1=(np.linalg.inv(future@lidar1)@global_point)[:3]
        label=source_label(x1,[present,future],[lidar0,lidar1],loader)
        ns=dict(curr_l2e_transform=lidar1,curr_e2g_transform=future,
                ref_g2e_transform=np.linalg.inv(present),ref_e2l_transform=np.linalg.inv(lidar0),
                ref_l2e_transform=lidar0,ref_e2g_transform=present,
                curr_g2e_transform=np.linalg.inv(future),curr_e2l_transform=np.linalg.inv(lidar1),
                total_cur2ref_lidar_transform=[],total_ref2cur_lidar_transform=[])
        exec(matrices,ns)
        future2ref=ns['cur_lidar_to_ref_lidar'];ref2future=ns['ref_lidar_to_cur_lidar']
        assert np.allclose(future2ref@ref2future,np.eye(4),atol=1e-12)
        # M0 first step: historical memory is reference t0, so ref_to_history=I.
        # Native uses [x,y,1,1]; planar yaw/translation avoids vertical ambiguity.
        env=dict(torch=types.SimpleNamespace(matmul=np.matmul),
                 future2ref=future2ref.reshape(1,1,4,4),ref_to_history_list=np.eye(4).reshape(1,1,4,4))
        exec(align,env)
        def query_samples(q):
            env['aligned_bev_coords']=np.array([q[0],q[1],1.,1.]).reshape(1,1,1,4)
            exec(point,env)
            return env['aligned_bev_coords'].reshape(4)[:2]
        future_query_sample=query_samples(x1)
        label_index_sample=query_samples(label)
        assert np.allclose(label,x0,atol=1e-12)
        assert np.allclose(future_query_sample,x0[:2],atol=1e-12)
        distance=float(np.linalg.norm(x1[:2]-label[:2]))
        if name=='identity':assert distance<1e-12
        else:assert distance>1.0 and np.linalg.norm(label_index_sample-x0[:2])>1.0
        # Changing native frame assignment to reference for the feature grid
        # means the consistent first-step geometric lookup would be identity.
        reference_consistent_sample=label[:2]
        assert np.allclose(reference_consistent_sample,x0[:2],atol=1e-12)
        cases.append(dict(case=name,world_point=global_point[:3].tolist(),
            future_ego_translation=translation,future_ego_yaw_radians=angle,
            future_local_point=x1.tolist(),source_loader_label_anchor_point=label.tolist(),
            native_query_of_stationary_point_xy=x1[:2].tolist(),
            native_memory_sample_for_that_query_xy=future_query_sample.tolist(),
            native_memory_sample_at_supervised_label_index_xy=label_index_sample.tolist(),
            query_label_displacement_m=distance,query_label_displacement_in_0p512m_cells=distance/.512,
            future2ref_row_matrix=future2ref.tolist(),ref2future_row_matrix=ref2future.tolist()))
    # Nontrivial fixed LiDAR extrinsic confirms conversion is not an identity
    # calibration artifact; no quantization/clipping is used in this proof.
    e0=transform(.3,[4.,-2.,.5]);e1=transform(-.5,[6.,3.,.7]);lidar=transform(.4,[1.,.2,1.5])
    world=np.array([10.,5.,1.,1.]);local=(np.linalg.inv(e1@lidar)@world)[:3]
    expected=(np.linalg.inv(e0@lidar)@world)[:3]
    actual=source_label(local,[e0,e1],[lidar,lidar],loader)
    assert np.allclose(actual,expected,atol=1e-12)
    output=dict(schema='m0-coordinate-contract-geometry-proof-v1',status='PASS_GEOMETRY_COUNTEREXAMPLES',
        proof_kind='Exact source-assignment AST with NumPy matrix arithmetic on synthetic physical planar poses; not a neural/CUDA or real-data inference test.',
        sources_sha256={str(path.relative_to(ROOT)):digest(path) for path in files.values()},
        executed_source_fragments=dict(loader=label_source,dataset_matrices=matrix_source,
                                      attention_matrix=align_source,attention_query=point_source),
        cases=cases,calibrated_extrinsic_label_error=float(np.max(np.abs(actual-expected))),
        established=['Loader expresses each sequence occupancy in current-anchor LiDAR coordinates.',
            'Declared query-to-memory correspondence uses the future-local grid, with stored memory frame transformed by ref2future.',
            'A zero-offset geometric correspondence places a stationary point at a different token than the anchor-frame target under nonidentity ego motion.',
            'Pointwise decoding, axis transpose and resize are not an ego-dependent coordinate warp.'],
        not_established=['Learned model outputs must remain future-local: offsets and action conditions can learn compensating mappings.',
            'Observed checkpoint performance loss is caused by this mismatch.',
            'Adding a post-hoc warp to a trained anchor-supervised model improves or preserves scores.',
            'Real cached target generation matches every inspected local helper on server; additional template/grid/encoder hashes should be verified.'],
        no_training=True,no_remote_calls=True)
    output['phantom_script_sha256']=digest(Path(__file__))
    OUT.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],cases=len(cases),output=str(OUT),sha256=digest(OUT))))


if __name__=='__main__':main()
