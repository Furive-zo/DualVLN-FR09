"""
InternVLAN1ForCausalLM.from_pretrained(..., ignore_mismatched_sizes=True)가
nn.Module이 아닌 raw nn.Parameter(latent_queries, memory_pos, rgb_resampler의
query_tokens/query_pos, rgb_model의 cls_token/pos_embed/LayerScale gamma,
traj_dit 블록의 gate 등)를 재초기화하지 않고 uninitialized 메모리(NaN/Inf)로
남기는 버그에 대한 방어 코드.

체크포인트 파일 자체가 정상이어도 from_pretrained를 호출할 때마다 매번
(임의의 조합으로) 재발하므로, 모델 로드 직후 항상 호출해야 한다.

중요: 이 raw Parameter들은 보통 체크포인트 안에 정상 학습된 값으로 저장돼 있다
(from_pretrained의 meta-tensor 복사가 no-op이 되어 반영이 안 될 뿐). 단,
`outputs/vla_e2e_s2_merged`처럼 merge 과정에서 이 버그에 안전장치 없이 걸려
NaN이 그대로 디스크에 저장돼버린 체크포인트도 존재한다 (이 경우는 로드해도
당연히 NaN). 그래서 복구 우선순위는:
  1순위: ckpt_dir (실제 로드 중인 체크포인트) 에서 동일 key를 읽되, 그 값 자체가
         NaN/Inf이면 무시한다.
  2순위: fallback_ckpt_dir (원본 DualVLN 베이스, 위 raw Parameter들이 실제로
         정상 값으로 확인된 소스) 에서 읽는다.
  3순위: model.rgb_model.* 는 원본 DepthAnythingV2 체크포인트에서 복원.
  4순위: 그래도 못 찾으면 정상 랜덤값으로 재초기화 (진짜로 어디서도 못 구한
         경우의 최후 수단 — 이 경로를 타면 해당 파라미터는 학습으로 복구해야 함).

버그 이력 (2026-07-28 수정): `reload_rgb_model_from_dav2()`는 원래 rgb_model
전체를 무조건 원본 DAv2 사전학습 가중치로 덮어썼다. 그런데 `train_s1_vla_e2e.py`의
`set_trainable()`을 보면 rgb_model은 S1 학습 대상(trainable_keys)에 포함돼 있어
실제로 파인튜닝된다 — 체크포인트 자체 safetensors에 그 파인튜닝된 값이 정상
저장돼 있는데도(checkpoint-2000 실측: 175개 중 52개가 원본 대비 상대오차 >1%,
평균 1.22%, 최대 25.65%) `reload_rgb_model_from_dav2`가 매 로드마다 그걸 지우고
사전학습 원본으로 되돌려버리고 있었다(실측으로 확인). 그래서 rgb_model도
`force_reload_s1_submodules`의 대상(S1_SUBMODULES)에 포함시켜 다른 서브모듈과
동일한 "체크포인트 자체 값 우선" 로직을 적용하도록 수정했다 — `reload_rgb_model_from_dav2`는
학습 시작 시점(아직 파인튜닝되지 않은 베이스 로드, train_s1_vla_e2e.py에서의 용도)에는
여전히 올바르므로 함수 자체는 남겨두되, 평가/추론 쪽 `load_model()`에서는 더 이상
호출하지 않는다.
"""
import json
import os

import torch

DAV2_CKPT = '/home/work/vln_project/vla_ws_restore/vla_project/checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
DEFAULT_FALLBACK_CKPT = '/home/work/.datasets/InternVLA-N1-DualVLN'


def _is_bad(t):
    tf = t.float()
    return torch.isnan(tf).any().item() or torch.isinf(tf).any().item()


def _load_ckpt_tensor(ckpt_dir, key):
    """ckpt_dir의 safetensors(단일 파일 또는 샤딩)에서 key 텐서를 읽어온다.
    없거나, 있어도 그 값 자체가 NaN/Inf이면 None을 반환한다."""
    if not ckpt_dir:
        return None
    from safetensors import safe_open

    index_path = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)['weight_map']
        shard = weight_map.get(key)
        if shard is None:
            return None
        with safe_open(os.path.join(ckpt_dir, shard), framework='pt', device='cpu') as f:
            t = f.get_tensor(key)
    else:
        single_path = os.path.join(ckpt_dir, 'model.safetensors')
        if not os.path.exists(single_path):
            return None
        with safe_open(single_path, framework='pt', device='cpu') as f:
            if key not in f.keys():
                return None
            t = f.get_tensor(key)

    return None if _is_bad(t) else t


def reload_rgb_model_from_dav2(model, verbose=True):
    """rgb_model 전체를 원본 DepthAnythingV2 체크포인트에서 통째로 재로드한다.

    build_depthanythingv2()가 InternVLAN1MetaModel.__init__ 안에서
    DAv2_model.load_state_dict(...)를 직접 호출하지만, 이 시점엔 HF
    from_pretrained의 meta-tensor 초기화 때문에 파라미터가 아직 meta 상태라
    그 load_state_dict 호출 자체가 통째로 no-op이 된다 ("copying from a
    non-meta parameter... to a meta parameter... no-op" 경고가 이거다).
    결과적으로 rgb_model의 거의 모든 파라미터(175개+)가 실제로는 로드된 적이
    없고, .cuda() 시점에 비로소 진짜 메모리로 잡히면서 각자 제멋대로 값이
    정해진다 — NaN도 있고(fix_nan_params가 잡음), 0이나 그럴듯한 랜덤값도
    있다(NaN이 아니라서 fix_nan_params가 못 잡음). 그래서 NaN 여부와 무관하게
    rgb_model 전체를 통째로 다시 로드해야 한다. 모델이 실제 device로 옮겨진
    (더는 meta가 아닌) 뒤에 호출해야 한다."""
    dav2 = torch.load(DAV2_CKPT, map_location='cpu')
    state = {k[len('pretrained.'):]: v for k, v in dav2.items() if k.startswith('pretrained.')}

    rgb_model = model.get_model().rgb_model
    target_dtype = next(rgb_model.parameters()).dtype
    state = {k: v.to(target_dtype) for k, v in state.items()}

    missing, unexpected = rgb_model.load_state_dict(state, strict=False)
    if verbose:
        print(f'[reload_rgb_model_from_dav2] loaded {len(state)} tensors '
              f'(missing={len(missing)}, unexpected={len(unexpected)})')
        if missing:
            print('  missing:', missing[:10], '...' if len(missing) > 10 else '')
        if unexpected:
            print('  unexpected:', unexpected[:10], '...' if len(unexpected) > 10 else '')


S1_SUBMODULES = ['rgb_model', 'cond_projector', 'action_encoder', 'action_decoder',
                  'memory_encoder', 'rgb_resampler', 'traj_dit']


def force_reload_s1_submodules(model, ckpt_dir, fallback_ckpt_dir=DEFAULT_FALLBACK_CKPT,
                                submodules=S1_SUBMODULES, verbose=True):
    """rgb_model과 같은 이유로, cond_projector/action_encoder/action_decoder/
    memory_encoder/rgb_resampler/traj_dit도 InternVLAN1MetaModel.__init__ 안에서
    동적으로 생성되는 표준 nn.Module인데, 체크포인트에 같은 이름의 값이 있어도
    HF from_pretrained의 meta-tensor materialize 과정에서 실제로 반영되지 않는
    경우가 있다 (파라미터 std가 매번 nn.Linear 기본 초기화 분포(~0.02)에 가깝게
    나오는 것으로 실측 확인됨 — NaN이 아니라서 fix_nan_params로는 못 잡는다).
    이 함수는 이 모듈들의 파라미터를 이름 단위로 체크포인트에서 강제로 다시
    읽어 덮어쓴다 (NaN 여부와 무관하게 항상 실행).

    rgb_model도 이 목록에 포함된다 — S1 학습 시 실제 파인튜닝 대상이라(
    set_trainable() 참고) 체크포인트 자체 값을 최우선으로 써야 하며, 예전처럼
    무조건 사전학습 DAv2 원본으로 되돌리면 파인튜닝이 사라진다. rgb_model만
    ckpt_dir/fallback_ckpt_dir 둘 다 실패할 경우의 3순위로 원본 DAv2 체크포인트를
    쓴다(예: `vla_e2e_s2_merged`처럼 rgb_model이 애초에 NaN으로 저장된 base)."""
    m = model.get_model()
    param_dict = dict(m.named_parameters())
    target = [n for n in param_dict if any(n == s or n.startswith(s + '.') for s in submodules)]

    dav2 = None
    from_ckpt, from_fallback, from_dav2, unresolved = [], [], [], []
    with torch.no_grad():
        for name in target:
            p = param_dict[name]
            full_key = 'model.' + name

            ref = _load_ckpt_tensor(ckpt_dir, full_key)
            if ref is not None and ref.shape == p.shape:
                p.copy_(ref.to(p.dtype))
                from_ckpt.append(name)
                continue

            ref = _load_ckpt_tensor(fallback_ckpt_dir, full_key)
            if ref is not None and ref.shape == p.shape:
                p.copy_(ref.to(p.dtype))
                from_fallback.append(name)
                continue

            if name.startswith('rgb_model.'):
                if dav2 is None:
                    dav2 = torch.load(DAV2_CKPT, map_location='cpu')
                dav2_key = 'pretrained.' + name[len('rgb_model.'):]
                dref = dav2.get(dav2_key)
                if dref is not None and dref.shape == p.shape and not _is_bad(dref):
                    p.copy_(dref.to(p.dtype))
                    from_dav2.append(name)
                    continue

            unresolved.append(name)

    if verbose:
        print(f'[force_reload_s1_submodules] {len(target)} params — '
              f'from ckpt: {len(from_ckpt)}, from fallback: {len(from_fallback)}, '
              f'from DAv2 ckpt: {len(from_dav2)}, unresolved: {len(unresolved)}')
        if unresolved:
            print('  unresolved:', unresolved[:10], '...' if len(unresolved) > 10 else '')
    return unresolved


def fix_nan_params(model, ckpt_dir=None, fallback_ckpt_dir=DEFAULT_FALLBACK_CKPT, verbose=True):
    """모델 로드 직후 호출: NaN/Inf인 raw parameter를 복구한다."""
    param_dict = dict(model.named_parameters())
    bad_names = [n for n, p in param_dict.items() if _is_bad(p)]
    if not bad_names:
        if verbose:
            print('[fix_nan_params] no NaN/Inf params found, nothing to do')
        return

    if verbose:
        print(f'[fix_nan_params] found {len(bad_names)} NaN/Inf params, repairing:')
        for n in bad_names:
            print('   ', n)

    dav2 = None
    from_ckpt, from_fallback, from_dav2, reinit, unresolved = [], [], [], [], []
    with torch.no_grad():
        for name in bad_names:
            p = param_dict[name]

            ref = _load_ckpt_tensor(ckpt_dir, name)
            if ref is not None and ref.shape == p.shape:
                p.copy_(ref.to(p.dtype))
                from_ckpt.append(name)
                continue

            ref = _load_ckpt_tensor(fallback_ckpt_dir, name)
            if ref is not None and ref.shape == p.shape:
                p.copy_(ref.to(p.dtype))
                from_fallback.append(name)
                continue

            if name.startswith('model.rgb_model.'):
                if dav2 is None:
                    dav2 = torch.load(DAV2_CKPT, map_location='cpu')
                dav2_key = 'pretrained.' + name[len('model.rgb_model.'):]
                dref = dav2.get(dav2_key)
                if dref is not None and dref.shape == p.shape and not _is_bad(dref):
                    p.copy_(dref.to(p.dtype))
                    from_dav2.append(name)
                    continue
                unresolved.append(name)
                continue

            p.copy_(torch.randn_like(p) * 0.02)
            reinit.append(name)

    if unresolved:
        raise RuntimeError(
            f'[fix_nan_params] could not resolve {len(unresolved)} bad params: {unresolved}'
        )

    still_bad = [n for n, p in param_dict.items() if _is_bad(p)]
    if still_bad:
        raise RuntimeError(f'[fix_nan_params] still NaN/Inf after repair: {still_bad}')

    if verbose:
        print(f'[fix_nan_params] restored from checkpoint: {len(from_ckpt)}, '
              f'from fallback (DualVLN base): {len(from_fallback)}, '
              f'from DAv2 ckpt: {len(from_dav2)}, re-initialized (random): {len(reinit)}')
