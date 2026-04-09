#!/usr/bin/env python3
"""
AI PBR 贴图生成器
================
输入颜色贴图 (Albedo/Diffuse)，自动生成完整PBR材质套装：
  - 法线图 (Normal Map)       — NormalBae AI / 深度推算
  - 高度图 (Height Map)       — Depth-Anything V2 AI
  - 粗糙度 (Roughness Map)    — 图像特征分析
  - 金属度 (Metallic Map)     — 颜色频谱分析
  - AO图   (Ambient Occlusion)— 多尺度高度遮蔽
  - MRAH   (合并贴图)         — R=Metallic G=Roughness B=AO A=Height

依赖安装: pip install -r requirements.txt
运行方式: python texture_generator.py
"""

import os
import sys
import traceback
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import numpy as np
import cv2
from PIL import Image
import torch
import gradio as gr

# ---- 修复 gradio_client 4.x 在 Python 3.9 上的 additionalProperties bug ----
try:
    import gradio_client.utils as _gc_utils
    _orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
    def _patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)
    _gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception:
    pass

# =====================================================================
# 全局模型实例
# =====================================================================
_depth_pipe = None       # Depth-Anything V2
_normal_detector = None  # NormalBae (controlnet_aux)
_zoom_js = ""

_device = "cuda" if torch.cuda.is_available() else "cpu"

MATERIAL_BASELINES = {
    "通用": {"normal_mul": 1.00, "roughness_offset": 0.00, "metallic_mul": 1.00},
    "木头": {"normal_mul": 1.30, "roughness_offset": 0.18, "metallic_mul": 0.15},
    "石头": {"normal_mul": 1.60, "roughness_offset": 0.22, "metallic_mul": 0.08},
    "地面": {"normal_mul": 1.50, "roughness_offset": 0.15, "metallic_mul": 0.12},
    "墙体": {"normal_mul": 1.40, "roughness_offset": 0.18, "metallic_mul": 0.10},
    "有机物": {"normal_mul": 0.95, "roughness_offset": 0.08, "metallic_mul": 0.05},
    "金属": {"normal_mul": 1.00, "roughness_offset": -0.12, "metallic_mul": 2.20},
}


# =====================================================================
# 设备信息
# =====================================================================
def get_device_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"GPU: {name}  ({vram:.1f} GB VRAM)"
    return "CPU  (无 GPU 加速，推理较慢)"


# =====================================================================
# 模型加载
# =====================================================================
def _load_depth(model_size: str):
    global _depth_pipe
    from transformers import pipeline as hf_pipeline

    model_map = {
        "Small (~100 MB)": "depth-anything/Depth-Anything-V2-Small-hf",
        "Base  (~390 MB)": "depth-anything/Depth-Anything-V2-Base-hf",
        "Large (~1.3 GB)": "depth-anything/Depth-Anything-V2-Large-hf",
    }
    model_id = model_map.get(model_size, model_map["Large (~1.3 GB)"])
    _depth_pipe = hf_pipeline(
        "depth-estimation",
        model=model_id,
        device=0 if _device == "cuda" else -1,
    )


def _load_normal():
    global _normal_detector
    try:
        from controlnet_aux import NormalBaeDetector
        _normal_detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        if _device == "cuda":
            _normal_detector = _normal_detector.to("cuda")
        return True
    except Exception:
        _normal_detector = None
        return False


def load_all_models(model_size: str, progress=gr.Progress()):
    progress(0.0, desc="初始化...")
    progress(0.5, desc=f"正在下载/加载深度模型 {model_size}（首次可能较慢）...")
    try:
        _load_depth(model_size)
    except Exception as e:
        return f"❌ 深度模型加载失败: {e}"

    progress(0.8, desc="加载法线估计模型 (NormalBae) ...")
    has_normal = _load_normal()

    progress(1.0, desc="完成!")
    normal_status = "✅ NormalBae" if has_normal else "⚠️  未加载 (将用深度推算代替)"
    return (
        f"✅ 模型加载完成\n"
        f"深度模型  : Depth-Anything V2 {model_size}\n"
        f"法线模型  : {normal_status}\n"
        f"运行设备  : {get_device_info()}"
    )


def apply_preset(mode: str):
    """返回预设参数，便于一键切换速度/质量模式。"""
    if mode == "fast":
        return (
            "Small (~100 MB)",
            "1024",
            True,    # one_click_clarity
            True,    # reference_enhance
            True,    # use_ai_normals
            4.0,     # normal_strength
            False,   # height_invert
            0.60,    # roughness_bias
            0.8,     # metallic_sensitivity
            "⚡ 已应用快速预设：Small模型 + 深度推算法线，适合先看结果。",
        )

    return (
        "Large (~1.3 GB)",
        "2048",
        True,    # one_click_clarity
        True,    # reference_enhance
        True,    # use_ai_normals
        5.0,     # normal_strength
        False,   # height_invert
        0.65,    # roughness_bias
        1.0,     # metallic_sensitivity
        "🎯 已应用高质量预设：Large模型 + NormalBae，首次加载较慢但质量更高。",
    )


def apply_material_baseline(
    material_type: str,
    normal_strength: float,
    roughness_bias: float,
    metallic_sensitivity: float,
):
    baseline = MATERIAL_BASELINES.get(material_type, MATERIAL_BASELINES["通用"])

    tuned_normal = float(np.clip(normal_strength * baseline["normal_mul"], 0.5, 10.0))
    tuned_rough = float(np.clip(roughness_bias + baseline["roughness_offset"], 0.0, 1.0))
    tuned_metal = float(np.clip(metallic_sensitivity * baseline["metallic_mul"], 0.0, 3.0))

    return tuned_normal, tuned_rough, tuned_metal


def parse_output_size(size_option: str, src_h: int, src_w: int):
    """解析输出尺寸选项，返回目标宽高。"""
    if size_option == "原始尺寸":
        return src_w, src_h

    target = int(size_option)
    return target, target


def enhance_clarity(img: np.ndarray, amount: float = 1.25, sigma: float = 1.0) -> np.ndarray:
    """使用反遮罩提升清晰度，支持灰度和RGB图。"""
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img_f, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_height_with_albedo_detail(
    pil_img: Image.Image,
    depth: np.ndarray,
    material_type: str,
) -> np.ndarray:
    """将颜色细节融合进高度图，提升砖缝/石缝等结构感。"""
    color = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 局部对比增强，让缝隙细节更明确
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0

    # 砖面亮度直接作为高度的主要参考（亮=高，缝隙暗=低）
    if material_type in ["墙体", "石头", "地面"]:
        depth_w, gray_w = 0.35, 0.65
    elif material_type == "木头":
        depth_w, gray_w = 0.45, 0.55
    else:
        depth_w, gray_w = 0.55, 0.45

    enhanced = depth * depth_w + gray_eq * gray_w
    enhanced = np.clip(enhanced, 0, 1)

    # 对比度拉伸
    p_low, p_high = np.percentile(enhanced, [1, 99])
    if p_high > p_low:
        enhanced = np.clip((enhanced - p_low) / (p_high - p_low), 0, 1)

    enhanced = cv2.bilateralFilter(enhanced.astype(np.float32), 9, 0.10, 9)
    return np.clip(enhanced, 0, 1)


def _safe_filename(name: str, fallback: str) -> str:
    """清理文件名并补全扩展名。"""
    raw = (name or "").strip()
    if not raw:
        raw = fallback
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        raw = raw.replace(ch, '_')
    if not raw.lower().endswith(".png"):
        raw += ".png"
    return raw


def export_selected_maps(
    normal_img,
    height_img,
    roughness_img,
    metallic_img,
    ao_img,
    mrah_rgba,
    export_dir: str,
    selected_maps,
    normal_name: str,
    height_name: str,
    roughness_name: str,
    metallic_name: str,
    ao_name: str,
    mrah_name: str,
):
    if not selected_maps:
        return "⚠️ 请先选择要导出的贴图。"

    out_dir = Path((export_dir or "").strip() or "./export")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_map = {
        "normal": (normal_img, normal_name, "normal"),
        "height": (height_img, height_name, "height"),
        "roughness": (roughness_img, roughness_name, "roughness"),
        "metallic": (metallic_img, metallic_name, "metallic"),
        "ao": (ao_img, ao_name, "ao"),
        "mrah": (mrah_rgba, mrah_name, "mrah"),
    }

    exported = []
    missing = []
    for key in selected_maps:
        img, custom_name, fallback = data_map.get(key, (None, "", key))
        if img is None:
            missing.append(key)
            continue

        file_name = _safe_filename(custom_name, fallback)
        Image.fromarray(np.array(img)).save(out_dir / file_name)
        exported.append(file_name)

    if not exported:
        return "❌ 没有可导出的贴图，请先点击“生成所有贴图”。"

    status = "✅ 导出完成！"
    status += f"\n目录: {out_dir.absolute()}"
    status += f"\n文件: {', '.join(exported)}"
    if missing:
        status += f"\n未导出(尚未生成): {', '.join(missing)}"
    return status


# =====================================================================
# 高度图 — Depth-Anything V2 AI 估计
# =====================================================================
def generate_height(
    pil_img: Image.Image,
    invert: bool,
    material_type: str,
    reference_enhance: bool,
) -> tuple:
    """返回 (height_clean, height_enhanced), 均为 float32 [0,1]
    height_clean: 纯 AI 深度（干净，供法线图用）
    height_enhanced: 融合颜色细节（供高度/AO/粗糙度用）
    """
    if _depth_pipe is None:
        raise RuntimeError("深度模型未加载，请先点击\"加载模型\"。")

    result = _depth_pipe(pil_img.convert("RGB"))
    depth_raw = result["depth"]

    # 兼容不同返回类型
    if isinstance(depth_raw, Image.Image):
        depth = np.array(depth_raw, dtype=np.float32)
    elif isinstance(depth_raw, torch.Tensor):
        depth = depth_raw.squeeze().cpu().numpy().astype(np.float32)
    else:
        depth = np.array(depth_raw, dtype=np.float32)

    # 归一化到 [0, 1]
    d_min, d_max = depth.min(), depth.max()
    depth = (depth - d_min) / (d_max - d_min + 1e-8) if d_max > d_min else np.zeros_like(depth)

    # 缩放到原始分辨率
    h, w = np.array(pil_img).shape[:2]
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

    # depth_clean: 原始 AI 深度（供法线用，Sobel 自然忽略全局渐变）
    depth_clean = depth.copy()
    if invert:
        depth_clean = 1.0 - depth_clean

    # 增强版：去除全局渐变 + 融合颜色细节（供高度/AO/粗糙度用）
    blur_radius = max(h, w) // 4
    if blur_radius % 2 == 0:
        blur_radius += 1
    blur_radius = max(blur_radius, 31)
    global_trend = cv2.GaussianBlur(depth, (blur_radius, blur_radius), blur_radius / 3.0)
    depth_local = depth - global_trend
    dl_min, dl_max = depth_local.min(), depth_local.max()
    if dl_max > dl_min:
        height_for_enhance = (depth_local - dl_min) / (dl_max - dl_min)
    else:
        height_for_enhance = np.zeros_like(depth_local)
    if invert:
        height_for_enhance = 1.0 - height_for_enhance

    if reference_enhance:
        height_enhanced = enhance_height_with_albedo_detail(pil_img, height_for_enhance, material_type)
    else:
        height_enhanced = height_for_enhance

    return depth_clean, height_enhanced


# =====================================================================
# 法线图
# =====================================================================
def generate_normal_from_height(height: np.ndarray, strength: float) -> np.ndarray:
    """由高度图计算法线图，返回 uint8 HxWx3 (OpenGL 切线空间)"""
    h32 = height.astype(np.float32)

    # 模糊去噪（sigma=3.5，柔化缝隙过渡，减少缝隙阴影）
    h32 = cv2.GaussianBlur(h32, (0, 0), 1.5)

    # Sobel 5 求梯度（Sobel 是高通滤波，自然忽略全局渐变）
    gx = cv2.Sobel(h32, cv2.CV_32F, 1, 0, ksize=5) * strength
    gy = cv2.Sobel(h32, cv2.CV_32F, 0, 1, ksize=5) * strength

    # 软压缩极端梯度，缝隙处法线偏转不超过阈值
    limit = 1.5
    gx = limit * np.tanh(gx / limit)
    gy = limit * np.tanh(gy / limit)

    nx = -gx
    ny =  gy   # OpenGL Y-up
    nz = np.ones_like(nx)

    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    length = np.maximum(length, 1e-8)
    nx /= length; ny /= length; nz /= length

    normal = np.stack([
        np.clip(nx * 0.5 + 0.5, 0, 1) * 255,
        np.clip(ny * 0.5 + 0.5, 0, 1) * 255,
        np.clip(nz * 0.5 + 0.5, 0, 1) * 255,
    ], axis=-1).astype(np.uint8)
    return normal


def generate_normal_ai(pil_img: Image.Image) -> np.ndarray:
    """NormalBae AI 法线估计，返回 uint8 HxWx3"""
    result = _normal_detector(pil_img.convert("RGB"))
    if isinstance(result, Image.Image):
        return np.array(result.convert("RGB"))
    return np.array(result)


# =====================================================================
# 粗糙度 — 多尺度频率 + 局部方差分析
# =====================================================================
def generate_roughness(color: np.ndarray, height: np.ndarray, bias: float) -> np.ndarray:
    """返回 float32 [0,1] 粗糙度图"""
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv  = cv2.cvtColor(color, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat  = hsv[:, :, 1] / 255.0
    val  = hsv[:, :, 2] / 255.0

    # 多尺度高频分量 → 细节越多越粗糙
    freq = np.zeros_like(gray)
    for sigma in [1, 2, 4, 8, 16]:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        freq += np.abs(gray - blurred)
    freq /= 5.0
    freq = np.clip(freq / (freq.max() + 1e-8), 0, 1)

    # 局部方差
    mean_sq  = cv2.GaussianBlur(gray ** 2, (21, 21), 5)
    sq_mean  = cv2.GaussianBlur(gray, (21, 21), 5) ** 2
    variance = np.sqrt(np.clip(mean_sq - sq_mean, 0, None))
    variance = np.clip(variance / (variance.max() + 1e-8), 0, 1)

    # 高度图梯度 → 缝隙/边缘位置更粗糙
    h_grad_x = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    h_grad_y = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    h_edge = np.sqrt(h_grad_x ** 2 + h_grad_y ** 2)
    h_edge = np.clip(h_edge / (h_edge.max() + 1e-8), 0, 1)

    roughness = (
        0.34 * freq +
        0.22 * variance +
        0.18 * (1.0 - val) +     # 暗区偏粗糙
        0.08 * (1.0 - sat) +     # 低饱和偏粗糙
        0.10 * (1.0 - height) +  # 凹陷处偏粗糙
        0.06 * h_edge +           # 缝隙边缘偏粗糙
        0.05                     # 基础偏移
    )
    roughness = np.clip(roughness + bias, 0, 1)
    roughness = cv2.bilateralFilter(roughness.astype(np.float32), 9, 0.12, 9)
    return np.clip(roughness, 0, 1)


# =====================================================================
# 金属度 — 颜色空间分析
# =====================================================================
def generate_metallic(color: np.ndarray, sensitivity: float) -> np.ndarray:
    """返回 float32 [0,1] 金属度图"""
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0

    # 消色差金属 (钢/铝/铬): 低饱和 + 高亮度（提高阈值避免非金属材质误检）
    gray_metal = np.power(np.maximum(0, 1.0 - sat * 2.5), 3) * np.power(np.maximum(0, val - 0.4), 2)

    # 黄金: H ≈ 18-38°
    gold = np.where(
        (hue >= 18) & (hue <= 38),
        np.minimum(sat * 1.5, 1.0) * np.minimum(val * 1.2, 1.0), 0.0
    )
    # 铜/青铜: H ≈ 8-20°
    copper = np.where(
        (hue >= 5) & (hue <= 20),
        np.minimum(sat * 1.2, 1.0) * np.minimum(val * 1.1, 1.0) * 0.75, 0.0
    )

    metallic = np.clip((gray_metal * 0.40 + gold + copper) * sensitivity, 0, 1)
    # 强力抑制低值噪声，非金属材质应接近纯黑
    metallic = np.where(metallic < 0.15, 0.0, metallic)
    metallic = cv2.GaussianBlur(metallic.astype(np.float32), (11, 11), 3.0)
    return np.clip(metallic, 0, 1)


# =====================================================================
# AO — 多尺度局部遮蔽
# =====================================================================
def generate_ao(height: np.ndarray) -> np.ndarray:
    """返回 float32 [0,1] AO 图"""
    h = height.astype(np.float32)
    # 先对高度做模糊，去除砖面微噪，只保留缝隙级别的高度变化
    h = cv2.GaussianBlur(h, (0, 0), 2.0)

    ao = np.ones(h.shape, dtype=np.float32)

    for radius, weight in [(3, 0.10), (7, 0.15), (15, 0.20), (31, 0.22), (63, 0.15)]:
        k = radius * 2 + 1
        local_mean = cv2.GaussianBlur(h, (k, k), radius / 3.0)
        shadow = np.clip((local_mean - h) * 2.5, 0, 1)
        ao -= shadow * weight

    # 曲率检测缝隙遮蔽（降低倍数避免噪点）
    lap = cv2.Laplacian(h, cv2.CV_32F, ksize=5)
    crevice = np.clip(-lap * 2.0, 0, 1)
    ao -= crevice * 0.15

    ao = np.clip(ao, 0, 1)
    ao = np.power(ao, 1.2)   # 轻微增强对比度
    # 较强的双边滤波清除残余噪点
    ao = cv2.bilateralFilter(ao, 9, 0.08, 9)
    return np.clip(ao, 0, 1)


# =====================================================================
# MRAH 合并贴图
# =====================================================================
def pack_mrah(
    metallic: np.ndarray,
    roughness: np.ndarray,
    ao: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    """打包为 uint8 HxWx4 RGBA (R=Metal G=Rough B=AO A=Height)"""
    return np.stack([
        (metallic  * 255).astype(np.uint8),
        (roughness * 255).astype(np.uint8),
        (ao        * 255).astype(np.uint8),
        (height    * 255).astype(np.uint8),
    ], axis=-1)


# =====================================================================
# 主处理函数
# =====================================================================
def process(
    input_image,
    material_type: str,
    output_size: str,
    one_click_clarity: bool,
    reference_enhance: bool,
    use_ai_normals: bool,
    normal_strength: float,
    height_invert: bool,
    roughness_bias: float,
    metallic_sensitivity: float,
    output_dir: str,
    save_outputs: bool,
    progress=gr.Progress(),
):
    if input_image is None:
        gr.Warning("请先上传颜色贴图！")
        return [None] * 6 + ["请先上传颜色贴图", None]

    if _depth_pipe is None:
        gr.Warning("模型尚未加载，请先点击“加载模型”。首次加载会下载模型，可能需要几分钟。")
        return [None] * 6 + ["模型尚未加载，请先点击“加载模型”。首次加载会下载模型，可能需要几分钟。", None]

    try:
        pil_img = Image.fromarray(input_image).convert("RGB")
        src_h, src_w = np.array(pil_img).shape[:2]
        dst_w, dst_h = parse_output_size(output_size, src_h, src_w)
        if (dst_w, dst_h) != (src_w, src_h):
            pil_img = pil_img.resize((dst_w, dst_h), Image.Resampling.LANCZOS)

        color_np = np.array(pil_img)
        tuned_normal_strength, tuned_roughness_bias, tuned_metallic_sensitivity = apply_material_baseline(
            material_type,
            normal_strength,
            roughness_bias,
            metallic_sensitivity,
        )

        # 1. 高度图 (AI)
        progress(0.05, desc="[1/6] AI 深度估计 → 高度图 ...")
        height_clean, height = generate_height(pil_img, height_invert, material_type, reference_enhance)

        # 2. 法线图（从干净的 AI 深度图计算，不含颜色纹理）
        progress(0.25, desc="[2/6] 生成法线图 ...")
        normal_from_height = generate_normal_from_height(height_clean, tuned_normal_strength)
        if use_ai_normals and _normal_detector is not None:
            normal_ai = generate_normal_ai(pil_img)
            # 对齐尺寸后混合 AI 法线(全局准确) 和 高度推算法线(边缘锐利)
            h_h, w_h = normal_from_height.shape[:2]
            if normal_ai.shape[:2] != (h_h, w_h):
                normal_ai = cv2.resize(normal_ai, (w_h, h_h), interpolation=cv2.INTER_CUBIC)
            normal = (normal_ai.astype(np.float32) * 0.5 + normal_from_height.astype(np.float32) * 0.5).astype(np.uint8)
        else:
            if use_ai_normals and _normal_detector is None:
                gr.Warning("NormalBae 未加载，改用高度深度推算法线")
            normal = normal_from_height

        # 3. 粗糙度
        progress(0.45, desc="[3/6] 生成粗糙度 ...")
        roughness = generate_roughness(color_np, height, tuned_roughness_bias - 0.5)

        # 4. 金属度
        progress(0.60, desc="[4/6] 生成金属度 ...")
        metallic = generate_metallic(color_np, tuned_metallic_sensitivity)

        # 5. AO
        progress(0.75, desc="[5/6] 生成 AO 图 ...")
        ao = generate_ao(height)

        # 6. MRAH
        progress(0.90, desc="[6/6] 打包 MRAH 合并贴图 ...")
        mrah = pack_mrah(metallic, roughness, ao, height)

        # 一键清晰增强（去模糊）
        if one_click_clarity:
            progress(0.95, desc="清晰增强处理中 ...")
            normal_u8 = normal  # 法线图不做锐化，避免放大噪声
            height_u8 = enhance_clarity((height * 255).astype(np.uint8), amount=1.1, sigma=0.9)
            rough_u8 = enhance_clarity((roughness * 255).astype(np.uint8), amount=1.0, sigma=1.0)
            metal_u8 = enhance_clarity((metallic * 255).astype(np.uint8), amount=0.8, sigma=1.0)
            ao_u8 = enhance_clarity((ao * 255).astype(np.uint8), amount=1.0, sigma=1.0)
            mrah = np.stack([metal_u8, rough_u8, ao_u8, height_u8], axis=-1)
        else:
            normal_u8 = normal
            height_u8 = (height * 255).astype(np.uint8)
            rough_u8 = (roughness * 255).astype(np.uint8)
            metal_u8 = (metallic * 255).astype(np.uint8)
            ao_u8 = (ao * 255).astype(np.uint8)

        # 保存
        if save_outputs and output_dir.strip():
            out_path = Path(output_dir.strip())
            out_path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(normal_u8).save(out_path / "normal.png")
            Image.fromarray(height_u8).save(out_path / "height.png")
            Image.fromarray(rough_u8).save(out_path / "roughness.png")
            Image.fromarray(metal_u8).save(out_path / "metallic.png")
            Image.fromarray(ao_u8).save(out_path / "ao.png")
            Image.fromarray(mrah).save(out_path / "mrah.png")

        progress(1.0, desc="完成!")
        status = (
            "✅ 生成完成！"
            f"\n材质基类: {material_type}"
            f"\n输出尺寸: {dst_w}x{dst_h}"
            f"\n清晰增强: {'开启' if one_click_clarity else '关闭'}"
            f"\n参考风格增强: {'开启' if reference_enhance else '关闭'}"
            f"\n生效参数: normal={tuned_normal_strength:.2f}, roughness={tuned_roughness_bias:.2f}, metallic={tuned_metallic_sensitivity:.2f}"
        )
        if save_outputs and output_dir.strip():
            status += f"\n已保存到: {out_path.absolute()}"

        # MRAH 以 RGB 展示 (Metal/Rough/AO)
        mrah_display = mrah[:, :, :3]

        return (
            normal_u8,
            height_u8,
            rough_u8,
            metal_u8,
            ao_u8,
            mrah_display,
            status,
            mrah,
        )

    except Exception as e:
        tb = traceback.format_exc()
        return [None] * 6 + [f"❌ 错误: {e}\n\n{tb}", None]


# =====================================================================
# Gradio 界面
# =====================================================================
_block_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {
    --bg-a: #f5f7ee;
    --bg-b: #e8efe0;
    --ink: #162218;
    --muted: #4f5d4f;
    --brand: #2d8f52;
    --brand-2: #1f6f6c;
    --card: rgba(255, 255, 255, 0.74);
    --stroke: rgba(34, 67, 43, 0.18);
}

.gradio-container {
    font-family: 'Space Grotesk', 'Noto Sans SC', sans-serif !important;
    color: var(--ink) !important;
    background:
        radial-gradient(1200px 420px at 10% -20%, #f6f0dc 20%, transparent 60%),
        radial-gradient(950px 360px at 100% -10%, #dfeee4 15%, transparent 58%),
        linear-gradient(160deg, var(--bg-a), var(--bg-b));
}

.hero-wrap {
    background: linear-gradient(120deg, rgba(255,255,255,0.65), rgba(232,246,235,0.72));
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 16px 18px 10px;
    margin-bottom: 10px;
    box-shadow: 0 10px 30px rgba(24, 44, 28, 0.08);
}

.hero-kicker {
    font-size: 12px;
    letter-spacing: 0.14em;
    color: var(--brand-2);
    font-weight: 700;
    margin-bottom: 6px;
}

.hero-title {
    margin: 0;
    font-size: 34px;
    line-height: 1.1;
    color: #19301f;
}

.hero-sub {
    margin: 8px 0 4px;
    color: var(--muted);
    font-size: 15px;
}

.panel {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 12px;
    backdrop-filter: blur(4px);
    box-shadow: 0 8px 20px rgba(24, 44, 28, 0.06);
}

.left-panel .gr-accordion {
    margin-bottom: 8px !important;
}

.left-panel .gr-tabs {
    margin-top: 2px !important;
}

.left-panel .gr-tab-nav {
    margin-bottom: 6px !important;
}

.left-panel .gr-form,
.left-panel .gr-box,
.left-panel .gr-block {
    gap: 8px !important;
}

.left-panel .gr-slider,
.left-panel .gr-checkbox,
.left-panel .gr-dropdown,
.left-panel .gr-textbox {
    margin-bottom: 6px !important;
}

.right-panel {
    min-height: auto;
}

.right-panel .gr-accordion {
    margin-top: 8px !important;
}

.output-img img {
    image-rendering: auto;
    border-radius: 12px;
    border: 1px solid rgba(26, 55, 36, 0.15);
    cursor: zoom-in;
}

.img-zoom-overlay {
    position: fixed;
    inset: 0;
    background: rgba(8, 14, 10, 0.86);
    backdrop-filter: blur(4px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 99999;
    flex-direction: column;
    padding: 18px;
}

.img-zoom-overlay.open {
    display: flex;
}

.img-zoom-target {
    max-width: min(92vw, 1600px);
    max-height: 86vh;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.26);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
}

.img-zoom-hint {
    margin-top: 10px;
    color: rgba(236, 248, 238, 0.9);
    font-size: 13px;
    letter-spacing: 0.02em;
}

button.primary {
    background: linear-gradient(135deg, var(--brand), #3aa464) !important;
    border: 0 !important;
}

footer { display: none !important; }
"""

def build_ui():
    global _zoom_js
    zoom_js = """
        () => {
            if (window.__pbrZoomInit) return;
            window.__pbrZoomInit = true;

            const overlay = document.createElement('div');
            overlay.className = 'img-zoom-overlay';
            overlay.innerHTML = '<img class="img-zoom-target" alt="zoom" /><div class="img-zoom-hint">双击预览已开启 · 单击任意处或按 Esc 关闭</div>';
            document.body.appendChild(overlay);

            const target = overlay.querySelector('.img-zoom-target');
            const closeZoom = () => overlay.classList.remove('open');
            const openZoom = (src) => {
                if (!src) return;
                target.src = src;
                overlay.classList.add('open');
            };

            overlay.addEventListener('click', closeZoom);
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') closeZoom();
            });

            const bindDblClick = () => {
                document.querySelectorAll('.output-img img').forEach((img) => {
                    if (img.dataset.zoomBound === '1') return;
                    img.dataset.zoomBound = '1';
                    img.addEventListener('dblclick', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        openZoom(img.currentSrc || img.src);
                    });
                });
            };

            const obs = new MutationObserver(bindDblClick);
            obs.observe(document.body, { childList: true, subtree: true });
            bindDblClick();
        }
        """
    _zoom_js = zoom_js

    if True:

        with gr.Blocks(title="AI PBR 贴图生成器", theme=gr.themes.Soft(), js=_zoom_js, css=_block_css) as demo:

            gr.Markdown(
                """
                <div class='hero-wrap'>
                  <div class='hero-kicker'>AI MATERIAL TOOLKIT</div>
                  <h1 class='hero-title'>AI PBR 贴图生成器</h1>
                  <p class='hero-sub'>从一张颜色贴图快速生成 Normal / Height / Roughness / Metallic / AO / MRAH</p>
                </div>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=360, elem_classes=["panel", "left-panel"]):
                    input_img = gr.Image(
                        label="颜色贴图 (Albedo / Diffuse)",
                        type="numpy",
                        height=240,
                    )

                    with gr.Accordion("模型与模式", open=True):
                        model_size_radio = gr.Dropdown(
                            choices=["Small (~100 MB)", "Base  (~390 MB)", "Large (~1.3 GB)"],
                            value="Small (~100 MB)",
                            label="深度模型规模",
                            info="建议先用 Small 验证流程，再切换 Large 提升质量",
                        )
                        device_display = gr.Textbox(
                            value=get_device_info(),
                            label="运行设备",
                            interactive=False,
                        )
                        with gr.Row():
                            preset_fast_btn = gr.Button("⚡ 快速预设", variant="secondary")
                            preset_quality_btn = gr.Button("🎯 高质量预设", variant="secondary")
                        load_btn = gr.Button("📥 加载模型", variant="primary")
                        model_status = gr.Textbox(
                            label="模型状态",
                            value="⚠️ 尚未加载模型，请先点击“加载模型”",
                            interactive=False,
                            lines=2,
                        )

                    with gr.Accordion("生成参数", open=True):
                        with gr.Tabs():
                            with gr.Tab("基础"):
                                with gr.Row():
                                    material_type = gr.Dropdown(
                                        choices=["通用", "木头", "石头", "地面", "墙体", "有机物", "金属"],
                                        value="通用",
                                        label="基础材质类型",
                                    )
                                    output_size = gr.Dropdown(
                                        choices=["原始尺寸", "256", "512", "1024", "2048", "4096"],
                                        value="1024",
                                        label="输出尺寸",
                                    )
                                with gr.Row():
                                    one_click_clarity = gr.Checkbox(
                                        label="一键清晰增强",
                                        value=True,
                                    )
                                    reference_enhance = gr.Checkbox(
                                        label="参考风格增强",
                                        value=True,
                                    )
                                    use_ai_normals = gr.Checkbox(
                                        label="使用 AI 法线",
                                        value=True,
                                    )
                                    height_invert = gr.Checkbox(
                                        label="反转高度图",
                                        value=False,
                                    )
                            with gr.Tab("高级"):
                                normal_strength = gr.Slider(
                                    0.5, 15.0, value=4.0, step=0.5,
                                    label="法线强度",
                                )
                                roughness_bias = gr.Slider(
                                    0.0, 1.0, value=0.60, step=0.05,
                                    label="粗糙度偏移",
                                )
                                metallic_sensitivity = gr.Slider(
                                    0.0, 3.0, value=1.0, step=0.1,
                                    label="金属度灵敏度",
                                )

                    generate_btn = gr.Button("🚀 生成所有贴图", variant="primary", size="lg")

                with gr.Column(scale=6, min_width=680, elem_classes=["panel", "right-panel"]):
                    status_box = gr.Textbox(label="生成状态", interactive=False, lines=4)

                    gr.Markdown("### 结果预览")
                    mrah_rgba_state = gr.State(value=None)
                    with gr.Row():
                        normal_out = gr.Image(label="Normal", height=210, elem_classes="output-img")
                        height_out = gr.Image(label="Height", height=210, elem_classes="output-img")
                        roughness_out = gr.Image(label="Roughness", height=210, elem_classes="output-img")
                    with gr.Row():
                        metallic_out = gr.Image(label="Metallic", height=210, elem_classes="output-img")
                        ao_out = gr.Image(label="AO", height=210, elem_classes="output-img")
                        mrah_out = gr.Image(label="MRAH (RGB预览)", height=210, elem_classes="output-img")

                    with gr.Accordion("💾 自动保存设置", open=True):
                        with gr.Row():
                            output_dir = gr.Textbox(
                                label="自动保存目录",
                                value=str(Path("./output").absolute()),
                                scale=5,
                            )
                            save_outputs = gr.Checkbox(label="自动保存", value=True, scale=1)

                    with gr.Accordion("📦 一键导出（可选贴图 + 自定义命名）", open=True):
                        export_selected = gr.CheckboxGroup(
                            choices=["normal", "height", "roughness", "metallic", "ao", "mrah"],
                            value=["normal", "height", "roughness", "metallic", "ao", "mrah"],
                            label="选择要导出的贴图",
                        )
                        export_dir = gr.Textbox(
                            label="导出目录",
                            value=str(Path("./export").absolute()),
                        )
                        with gr.Row():
                            export_name_normal = gr.Textbox(label="normal 文件名", value="normal")
                            export_name_height = gr.Textbox(label="height 文件名", value="height")
                            export_name_roughness = gr.Textbox(label="roughness 文件名", value="roughness")
                        with gr.Row():
                            export_name_metallic = gr.Textbox(label="metallic 文件名", value="metallic")
                            export_name_ao = gr.Textbox(label="ao 文件名", value="ao")
                            export_name_mrah = gr.Textbox(label="mrah 文件名", value="mrah")
                        export_btn = gr.Button("📦 导出所选贴图", variant="secondary")
                        export_status = gr.Textbox(label="导出状态", interactive=False, lines=3)

            # ── 事件绑定 ──────────────────────────────────────
            load_btn.click(
                fn=load_all_models,
                inputs=[model_size_radio],
                outputs=[model_status],
            )

            preset_fast_btn.click(
                fn=lambda: apply_preset("fast"),
                inputs=[],
                outputs=[
                    model_size_radio,
                    output_size,
                    one_click_clarity,
                    reference_enhance,
                    use_ai_normals,
                    normal_strength,
                    height_invert,
                    roughness_bias,
                    metallic_sensitivity,
                    status_box,
                ],
            )

            preset_quality_btn.click(
                fn=lambda: apply_preset("quality"),
                inputs=[],
                outputs=[
                    model_size_radio,
                    output_size,
                    one_click_clarity,
                    reference_enhance,
                    use_ai_normals,
                    normal_strength,
                    height_invert,
                    roughness_bias,
                    metallic_sensitivity,
                    status_box,
                ],
            )

            generate_btn.click(
                fn=process,
                inputs=[
                    input_img, material_type, output_size, one_click_clarity,
                    reference_enhance,
                    use_ai_normals, normal_strength, height_invert,
                    roughness_bias, metallic_sensitivity, output_dir, save_outputs,
                ],
                outputs=[
                    normal_out, height_out, roughness_out,
                    metallic_out, ao_out, mrah_out,
                    status_box,
                    mrah_rgba_state,
                ],
            )

            export_btn.click(
                fn=export_selected_maps,
                inputs=[
                    normal_out, height_out, roughness_out, metallic_out, ao_out, mrah_rgba_state,
                    export_dir, export_selected,
                    export_name_normal, export_name_height, export_name_roughness,
                    export_name_metallic, export_name_ao, export_name_mrah,
                ],
                outputs=[export_status],
            )

        # 开启请求队列后，长时间推理会显示排队/运行状态，避免“无响应”观感
        demo.queue(default_concurrency_limit=1, max_size=8)
        return demo


# =====================================================================
# 启动入口
# =====================================================================
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
