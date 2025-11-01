#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URPC2021 Marine RT-DETR 全流程脚本
================================
单个 Python 文件串联以下功能：
- 数据集 YAML 生成（URPC/通用类名自由切换）
- COCO/VOC → YOLO 转换（含 URPC 到通用类名映射）
- 训练/验证拆分
- UCCP 水下图像预处理（灰度世界 WB + LAB-CLAHE + 轻度去噪）
- RT-DETR 训练、验证指标导出
- 推理可视化（含 Gamma-TTA + 分类别 NMS 融合）
- 论文 Word 骨架导出（可自动填入指标）

推荐用法（PowerShell）：
    python urpc21_all_in_one.py write-yaml --root "C:/Users/Ka wing Ho/Desktop/人工智能/URPC2021"
    python urpc21_all_in_one.py train --data underwater_urpc21_win.yaml --model rtdetr-l.pt --device 0 --workers 0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from types import SimpleNamespace

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# 默认数据根目录，CLI 未指定 --root 时退回到仓库同级的 URPC2021
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "URPC2021"
IM_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASSES_CANONICAL = ["sea_cucumber", "sea_urchin", "scallop", "starfish"]
CANONICAL_TO_ID = {name: idx for idx, name in enumerate(CLASSES_CANONICAL)}
CANONICAL_ALIAS = {name.lower(): name for name in CLASSES_CANONICAL}
URPC21_ALIAS = {
    "holothurian": "sea_cucumber",
    "echinus": "sea_urchin",
    "scallop": "scallop",
    "starfish": "starfish",
}

YAML_CANONICAL = """\
path: "{root}"
train: images/train
val: images/val
names:
  0: sea_cucumber
  1: sea_urchin
  2: scallop
  3: starfish
"""

YAML_URPC21 = """\
path: "{root}"
train: images/train
val: images/val
names:
  0: holothurian
  1: echinus
  2: scallop
  3: starfish
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def info(message: str) -> None:
    """统一打印格式，方便定位脚本输出。"""
    print(f"[URPC21] {message}")


def timestamp() -> str:
    """返回当前时间戳（用于输出目录命名）。"""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    """确保目录存在，若不存在则递归创建。"""
    path.mkdir(parents=True, exist_ok=True)


def to_posix(path: Path) -> str:
    """转换为正斜杠路径，避免 Windows 下 YAML 解析问题。"""
    return path.as_posix()


def check_exists(path: Path, what: str) -> Path:
    """校验路径存在，不存在时给出更友好的错误信息。"""
    if not path.exists():
        raise FileNotFoundError(f"{what} 不存在：{path}")
    return path


def normalize_label(raw: Optional[str], urpc21: bool) -> Optional[str]:
    """对标签名做清洗/映射，支持 URPC21 -> 通用类名。"""
    if not raw:
        return None
    key = raw.strip().lower()
    if urpc21:
        mapped = URPC21_ALIAS.get(key)
        if mapped:
            return mapped
    mapped = CANONICAL_ALIAS.get(key)
    if mapped:
        return mapped
    return None


def resolve_image(images_dir: Path, file_name: str) -> Optional[Path]:
    """根据 file_name 在目录中查找实际存在的图像，兼容不同扩展名。"""
    candidate = images_dir / Path(file_name)
    if candidate.exists():
        return candidate
    stem = Path(file_name).stem
    for ext in IM_EXTS:
        alt = images_dir / f"{stem}{ext}"
        if alt.exists():
            return alt
    for ext in IM_EXTS:
        hits = list(images_dir.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def copy_image_to_dir(src: Path, dst_dir: Path) -> Path:
    """将图像复制到目标目录，避免重复写入同一文件。"""
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    try:
        same = src.resolve() == dst.resolve()
    except FileNotFoundError:
        same = False
    if not same:
        shutil.copy2(src, dst)
    return dst


def write_yolo_labels(path: Path, rows: Sequence[Sequence[float]]) -> None:
    """把 YOLO 归一化框列表写入 txt 标签文件。"""
    ensure_dir(path.parent)
    if rows:
        lines = [
            f"{int(cid)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            for cid, xc, yc, w, h in rows
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """使用 numpy + imdecode 读取图像，兼容中文/空格路径。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """使用 imencode + tofile 写图像，避免 cv2.imwrite 在中文路径下失败。"""
    ext = Path(path).suffix or ".jpg"
    try:
        success, encoded = cv2.imencode(ext, img)
        if not success:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1) YAML writer
# ---------------------------------------------------------------------------

def cmd_write_yaml(args: argparse.Namespace) -> None:
    """CLI：生成数据集 YAML，打印内容以便用户确认。"""
    root = Path(args.root).expanduser().resolve()
    content = (YAML_URPC21 if args.urpc21 else YAML_CANONICAL).format(
        root=to_posix(root)
    )
    outfile = Path(args.outfile)
    ensure_dir(outfile.parent if outfile.parent != Path("") else Path("."))
    outfile.write_text(content, encoding="utf-8")
    info(f"数据集 YAML 已写入：{outfile}")
    print(content)


# ---------------------------------------------------------------------------
# 2) COCO/VOC → YOLO 转换
# ---------------------------------------------------------------------------

def convert_coco_to_yolo(
    images_dir: Path, ann_path: Path, out_dir: Path, urpc21: bool
) -> None:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_name = {int(c["id"]): c["name"] for c in data.get("categories", [])}
    id_to_image = {int(img["id"]): img for img in data.get("images", [])}
    grouped: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        grouped.setdefault(int(ann["image_id"]), []).append(ann)

    out_img_dir = out_dir / "images" / "train"
    out_lbl_dir = out_dir / "labels" / "train"
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    copied = 0
    skipped = 0
    for img_id, meta in id_to_image.items():
        file_name = meta.get("file_name")
        width = meta.get("width")
        height = meta.get("height")
        if not file_name or not width or not height:
            skipped += 1
            continue
        src = resolve_image(images_dir, file_name)
        if not src:
            skipped += 1
            continue
        dst_img = copy_image_to_dir(src, out_img_dir)
        anns = grouped.get(img_id, [])
        rows = []
        for ann in anns:
            label_name = normalize_label(
                id_to_name.get(int(ann.get("category_id", -1))), urpc21
            )
            if label_name is None:
                continue
            cid = CANONICAL_TO_ID[label_name]
            x, y, w, h = ann["bbox"]
            xc = (x + w / 2) / width
            yc = (y + h / 2) / height
            bw = w / width
            bh = h / height
            rows.append((cid, xc, yc, bw, bh))
        write_yolo_labels(out_lbl_dir / f"{dst_img.stem}.txt", rows)
        copied += 1
    info(f"COCO → YOLO 转换完成：{copied} 张图像，{skipped} 张跳过。")


def convert_voc_to_yolo(
    images_dir: Path, ann_dir: Path, out_dir: Path, urpc21: bool
) -> None:
    import xml.etree.ElementTree as ET

    xml_files = sorted(ann_dir.glob("*.xml"))
    out_img_dir = out_dir / "images" / "train"
    out_lbl_dir = out_dir / "labels" / "train"
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    converted = 0
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.findtext("filename")
        size = root.find("size")
        if filename is None or size is None:
            continue
        width = float(size.findtext("width", default="0") or 0)
        height = float(size.findtext("height", default="0") or 0)
        src = resolve_image(images_dir, filename)
        if not src or width <= 0 or height <= 0:
            continue
        dst_img = copy_image_to_dir(src, out_img_dir)
        rows = []
        for obj in root.findall("object"):
            raw = obj.findtext("name")
            label_name = normalize_label(raw, urpc21)
            if label_name is None:
                continue
            cid = CANONICAL_TO_ID[label_name]
            b = obj.find("bndbox")
            if b is None:
                continue
            xmin = float(b.findtext("xmin", default="0") or 0)
            ymin = float(b.findtext("ymin", default="0") or 0)
            xmax = float(b.findtext("xmax", default="0") or 0)
            ymax = float(b.findtext("ymax", default="0") or 0)
            xc = ((xmin + xmax) / 2) / width
            yc = ((ymin + ymax) / 2) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            rows.append((cid, xc, yc, bw, bh))
        write_yolo_labels(out_lbl_dir / f"{dst_img.stem}.txt", rows)
        converted += 1
    info(f"VOC → YOLO 转换完成：{converted} 张图像。")


def cmd_convert(args: argparse.Namespace) -> None:
    """CLI：COCO/VOC 数据转 YOLO，支持 URPC21 标签映射。"""
    images_dir = check_exists(Path(args.images), "images 目录")
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    if args.fmt == "coco":
        ann_path = check_exists(Path(args.ann), "COCO 标注文件")
        convert_coco_to_yolo(images_dir, ann_path, out_dir, args.urpc21)
    else:
        ann_dir = check_exists(Path(args.ann), "VOC 标注目录")
        convert_voc_to_yolo(images_dir, ann_dir, out_dir, args.urpc21)


# ---------------------------------------------------------------------------
# 3) Train/Val split
# ---------------------------------------------------------------------------

def cmd_split(args: argparse.Namespace) -> None:
    """CLI：从训练集中随机划分部分样本到验证集。"""
    root = Path(args.root).resolve()
    train_img_dir = check_exists(root / "images" / "train", "images/train")
    train_lbl_dir = check_exists(root / "labels" / "train", "labels/train")
    val_img_dir = root / "images" / "val"
    val_lbl_dir = root / "labels" / "val"
    ensure_dir(val_img_dir)
    ensure_dir(val_lbl_dir)

    images = sorted(
        [p for p in train_img_dir.iterdir() if p.suffix.lower() in IM_EXTS]
    )
    if not images:
        info("?? images/train ?????????")
        return

    random.seed(args.seed)
    random.shuffle(images)
    n_val = int(len(images) * args.ratio)
    if args.ratio > 0 and n_val == 0:
        n_val = 1
    selected = images[:n_val]

    moved = 0
    for img_path in selected:
        label_path = train_lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        target_img = val_img_dir / img_path.name
        target_lbl = val_lbl_dir / label_path.name
        if target_img.exists() or target_lbl.exists():
            continue
        shutil.move(str(img_path), str(target_img))
        shutil.move(str(label_path), str(target_lbl))
        moved += 1
    info(f"??????? {moved} ???/??? val??? {args.ratio:.2f}?")


# ---------------------------------------------------------------------------
# 4) UCCP ???
# ---------------------------------------------------------------------------

def gray_world_wb(img: np.ndarray) -> np.ndarray:
    """实现灰度世界假设白平衡。"""
    img32 = img.astype(np.float32)
    mean = img32.mean(axis=(0, 1))
    gray = float(mean.mean())
    scale = gray / (mean + 1e-6)
    corrected = np.clip(img32 * scale, 0, 255).astype(np.uint8)
    return corrected


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """在 LAB 空间的 L 通道执行自适应直方图均衡。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def mild_denoise(img: np.ndarray) -> np.ndarray:
    """使用 fastNlMeans 进行轻度彩色去噪。"""
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )


def apply_uccp(img: np.ndarray) -> np.ndarray:
    out = gray_world_wb(img)
    out = apply_clahe(out)
    out = mild_denoise(out)
    return out


def _uccp_worker(args) -> bool:
    src_path, dst_path, skip_existing = args
    dst_path = Path(dst_path)
    if skip_existing and dst_path.exists():
        return False
    img = imread_unicode(str(src_path))
    if img is None:
        return False
    ensure_dir(dst_path.parent)
    processed = apply_uccp(img)
    return imwrite_unicode(str(dst_path), processed)


def cmd_preprocess(args: argparse.Namespace) -> None:
    src = check_exists(Path(args.src), "输入目录")
    dst = Path(args.dst)
    ensure_dir(dst)
    files = sorted(
        p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IM_EXTS
    )
    if not files:
        info("未找到可处理的图像。")
        return
    from tqdm import tqdm

    tasks = []
    for img_path in files:
        try:
            rel = img_path.relative_to(src)
        except ValueError:
            rel = Path(img_path.name)
        out_path = dst / rel
        tasks.append((str(img_path), str(out_path), args.skip_existing))

    jobs = args.jobs
    if jobs is None:
        jobs = 0

    processed = 0
    if jobs == 1:
        iterator = map(_uccp_worker, tasks)
        desc = "UCCP"
        for result in tqdm(iterator, total=len(tasks), desc=desc, unit="img"):
            processed += int(result)
    else:
        if jobs <= 0:
            max_workers = os.cpu_count() or 1
        else:
            max_workers = jobs

        if max_workers <= 1:
            iterator = map(_uccp_worker, tasks)
            desc = "UCCP"
            for result in tqdm(iterator, total=len(tasks), desc=desc, unit="img"):
                processed += int(result)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                for result in tqdm(
                    executor.map(_uccp_worker, tasks),
                    total=len(tasks),
                    desc=f"UCCP x{max_workers}",
                    unit="img",
                ):
                    processed += int(result)

    info(
        f"UCCP 处理完成：{processed} 张更新，目标目录 {dst}（总计 {len(tasks)} 张）"
    )


# ---------------------------------------------------------------------------
# 5) ?? / 6) ?? / 7) ??
# ---------------------------------------------------------------------------

def require_ultralytics() -> None:
    """若未安装 ultralytics，给出友好错误提示。"""
    if YOLO is None:
        raise RuntimeError("Ultralytics 未安装，请先运行：pip install ultralytics")


def cmd_train(args: argparse.Namespace) -> None:
    """CLI：调用 Ultralytics YOLO 的 train 接口训练 RT-DETR。"""
    require_ultralytics()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device,
        seed=args.seed,
        optimizer="adamw",
        lr0=args.lr0,
        cos_lr=True,
        mosaic=0.7,
        mixup=0.1,
    )
    info("?????????? runs/detect/<name>/weights/")


def safe_float(value: Optional[float]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cmd_val(args: argparse.Namespace) -> None:
    """CLI：执行验证评估并导出 JSON/CSV 指标。"""
    require_ultralytics()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        augment=getattr(args, "augment", False),
        plots=args.plots,
        verbose=True,
    )
    save_dir = Path(args.save_dir or Path("runs") / "val_export" / timestamp())
    ensure_dir(save_dir)
    names = getattr(model, "names", None)
    box_metrics = getattr(metrics, "box", None)
    speed_metrics = getattr(metrics, "speed", None)

    summary: Dict[str, object] = {}
    if box_metrics is not None:
        summary["map50_95"] = safe_float(getattr(box_metrics, "map", None))
        summary["map50"] = safe_float(getattr(box_metrics, "map50", None))
        summary["map75"] = safe_float(getattr(box_metrics, "map75", None))
        summary["precision_mean"] = safe_float(getattr(box_metrics, "mp", None))
        summary["recall_mean"] = safe_float(getattr(box_metrics, "mr", None))
        maps = getattr(box_metrics, "maps", None)
        if maps is not None and names is not None:
            per_class = {
                str(names.get(idx, idx)): safe_float(ap)
                for idx, ap in enumerate(maps)
            }
            summary["per_class_ap50_95"] = per_class
            csv_path = save_dir / "per_class_ap50_95.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(["class", "AP50_95"])
                for cls_name, score in per_class.items():
                    writer.writerow(
                        [cls_name, f"{score:.6f}" if score is not None else ""]
                    )
    if speed_metrics:
        try:
            summary["speed_ms"] = {
                k: float(v) for k, v in speed_metrics.items()
            }
        except Exception:
            pass

    json_path = save_dir / "val_metrics.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    info(f"????????{json_path}")
    if args.plots:
        info("Ultralytics ??? PR ???????? runs/detect/val*")


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma 校正，用于多曝光推理增强。"""
    img_norm = np.clip(img.astype(np.float32) / 255.0, 0, 1)
    corrected = np.power(img_norm, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两组框的 IoU，用于 NMS 阶段。"""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def per_class_nms(
    boxes: np.ndarray, scores: np.ndarray, clses: np.ndarray, iou_thr: float
) -> np.ndarray:
    """对每个类别独立执行 NMS，避免跨类抑制。"""
    if boxes.size == 0:
        return np.empty(0, dtype=int)
    keep: List[int] = []
    classes = clses.astype(int)
    for cls_id in np.unique(classes):
        cls_idx = np.where(classes == cls_id)[0]
        cls_boxes = boxes[cls_idx]
        cls_scores = scores[cls_idx]
        order = np.argsort(-cls_scores)
        while order.size:
            current = order[0]
            keep.append(cls_idx[current])
            if order.size == 1:
                break
            remaining = order[1:]
            ious = iou_xyxy(cls_boxes[current : current + 1], cls_boxes[remaining])[0]
            order = remaining[ious <= iou_thr]
    return np.array(keep, dtype=int)


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    clses: np.ndarray,
    names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """在图像上绘制检测框（用于推理可视化和 TTA 输出）。"""
    palette = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 128),
        (128, 0, 255),
        (0, 255, 255),
    ]
    out = img.copy()
    for box, score, cls in zip(boxes, scores, clses):
        color = palette[int(cls) % len(palette)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = str(int(cls))
        if names and int(cls) in names:
            label = str(names[int(cls)])
        text = f"{label}:{score:.2f}"
        cv2.putText(
            out,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return out


def gather_source_images(source: Path) -> List[Path]:
    """收集需要推理的图片列表，可传入目录或单个文件。"""
    if source.is_dir():
        return sorted(
            [p for p in source.iterdir() if p.suffix.lower() in IM_EXTS]
        )
    if source.exists():
        return [source]
    return []


def parse_gamma_values(raw: str) -> List[float]:
    """解析命令行传入的 gamma 字符串，过滤非法值。"""
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = float(chunk)
            if value > 0:
                values.append(value)
        except ValueError:
            continue
    return values or [0.8, 1.0, 1.2]


def cmd_infer(args: argparse.Namespace) -> None:
    """CLI：执行常规或 Gamma-TTA 推理并保存可视化结果。"""
    require_ultralytics()
    model = YOLO(args.weights)
    names = getattr(model, "names", None)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    images = gather_source_images(Path(args.source))
    if not images:
        info("??????????")
        return
    gamma_values = parse_gamma_values(args.gamma_values) if args.tta else [1.0]

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if not args.tta:
            result = model.predict(
                source=img[..., ::-1],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            rendered = result.plot()[:, :, ::-1]
            cv2.imwrite(str(save_dir / img_path.name), rendered)
            continue

        boxes_list: List[np.ndarray] = []
        scores_list: List[np.ndarray] = []
        cls_list: List[np.ndarray] = []
        for gamma_value in gamma_values:
            enhanced = gamma_correct(img, gamma_value)
            result = model.predict(
                source=enhanced[..., ::-1],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes_list.append(result.boxes.xyxy.cpu().numpy())
            scores_list.append(result.boxes.conf.cpu().numpy())
            cls_list.append(result.boxes.cls.cpu().numpy())

        if not boxes_list:
            fallback = model.predict(
                source=img[..., ::-1],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            rendered = fallback.plot()[:, :, ::-1]
            cv2.imwrite(str(save_dir / img_path.name), rendered)
            continue

        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        clses = np.concatenate(cls_list, axis=0)
        keep = per_class_nms(boxes, scores, clses, iou_thr=args.tta_iou)
        merged = draw_boxes(img, boxes[keep], scores[keep], clses[keep], names=names)
        cv2.imwrite(str(save_dir / img_path.name), merged)
    info(f"推理结果已保存至：{save_dir}")


# ---------------------------------------------------------------------------
# 8) 一键流水线（Pipeline）
# ---------------------------------------------------------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    """CLI：按需串联写 YAML、UCCP、训练、验证与推理的全流程。"""
    dataset_root = check_exists(
        Path(args.root).expanduser().resolve(),
        "数据集根目录",
    )
    preprocess_jobs = args.preprocess_jobs
    preprocess_skip = args.preprocess_skip_existing

    yaml_path = (
        Path(args.data).expanduser().resolve()
        if args.data
        else (Path.cwd() / "underwater_urpc21_win.yaml")
    )
    if args.write_yaml or not yaml_path.exists():
        info("运行 YAML 生成步骤。")
        ns = SimpleNamespace(
            root=str(dataset_root),
            outfile=str(yaml_path),
            urpc21=args.urpc21,
        )
        cmd_write_yaml(ns)

    current_yaml = yaml_path

    use_uccp = args.use_uccp or args.preprocess

    if args.preprocess:
        info("执行 UCCP 预处理（train/val）...")
        for split in ("train", "val"):
            src = dataset_root / "images" / split
            check_exists(src, f"{split} 图像目录")
            dst = dataset_root / "images" / f"{split}_uccp"
            ns = SimpleNamespace(
                src=str(src),
                dst=str(dst),
                jobs=preprocess_jobs,
                skip_existing=preprocess_skip,
            )
            cmd_preprocess(ns)

    if use_uccp:
        yaml_uccp = yaml_path.with_name(yaml_path.stem + "_uccp.yaml")
        content = yaml_path.read_text(encoding="utf-8")
        content = content.replace("images/train", "images/train_uccp")
        content = content.replace("images/val", "images/val_uccp")
        yaml_uccp.write_text(content, encoding="utf-8")
        current_yaml = yaml_uccp
        info(f"已生成 UCCP 版 YAML：{yaml_uccp}")
    else:
        current_yaml = yaml_path

    project_dir = Path(args.project)
    weights_path = (
        Path(args.weights).expanduser().resolve()
        if args.weights
        else (project_dir / args.name / "weights" / "best.pt")
    )

    if not args.skip_train:
        info("开始训练...")
        train_ns = SimpleNamespace(
            data=str(current_yaml),
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=args.project,
            name=args.name,
            device=args.device,
            seed=args.seed,
            lr0=args.lr0,
        )
        cmd_train(train_ns)
        weights_path = weights_path.resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"未找到权重文件：{weights_path}")

    if not args.skip_val:
        info("开始验证...")
        save_dir = (
            str(Path(args.val_save_dir).expanduser().resolve())
            if args.val_save_dir
            else None
        )
        val_ns = SimpleNamespace(
            data=str(current_yaml),
            weights=str(weights_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=args.plots,
            save_dir=save_dir,
        )
        cmd_val(val_ns)

    if not args.skip_infer:
        info("开始推理可视化...")
        source = (
            Path(args.infer_source).expanduser().resolve()
            if args.infer_source
            else dataset_root / "images" / ("val_uccp" if use_uccp else "val")
        )
        infer_ns = SimpleNamespace(
            weights=str(weights_path),
            source=str(source),
            save_dir=str(Path(args.infer_out).expanduser().resolve()),
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            tta=args.tta,
            gamma_values=args.gamma_values,
            tta_iou=args.tta_iou,
        )
        cmd_infer(infer_ns)

    info("Pipeline 全流程完成。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    """构造命令行解析器并注册所有子命令。"""
    ap = argparse.ArgumentParser(
        description="URPC2021 Marine RT-DETR All-in-One ????"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    default_root = to_posix(DEFAULT_DATA_ROOT)

    w = sub.add_parser("write-yaml", help="????? YAML?Windows ?????")
    w.add_argument(
        "--root",
        default=default_root,
        help=f"??????????{default_root}?",
    )
    w.add_argument(
        "--outfile", default="underwater_urpc21_win.yaml", help="YAML ????"
    )
    w.add_argument(
        "--urpc21", action="store_true", help="?? URPC2021 ???holothurian ??"
    )
    w.set_defaults(func=cmd_write_yaml)

    c = sub.add_parser("convert", help="COCO/VOC ? YOLO ??")
    c.add_argument("--fmt", choices=["coco", "voc"], required=True)
    c.add_argument("--images", required=True, help="??????")
    c.add_argument(
        "--ann", required=True, help="COCO JSON ? VOC XML ??/??"
    )
    c.add_argument("--out", default="data", help="?? YOLO ?????")
    c.add_argument(
        "--urpc21",
        action="store_true",
        help="? holothurian/echinus ???????",
    )
    c.set_defaults(func=cmd_convert)

    s = sub.add_parser("split", help="?????????")
    s.add_argument(
        "--root", required=True, help="YOLO ????????? images/labels?"
    )
    s.add_argument("--ratio", type=float, default=0.2, help="val ??")
    s.add_argument("--seed", type=int, default=42, help="????")
    s.set_defaults(func=cmd_split)

    p = sub.add_parser("preprocess", help="UCCP 预处理整个图像目录")
    p.add_argument("--src", required=True, help="原始图像目录")
    p.add_argument("--dst", required=True, help="输出目录（建议 *_uccp）")
    p.add_argument("--jobs", type=int, default=0, help="并行进程数（0 自动，1 单进程）")
    p.add_argument("--skip-existing", action="store_true", help="若输出已存在则跳过")
    p.set_defaults(func=cmd_preprocess)

    t = sub.add_parser("train", help="?? RT-DETR??? Ultralytics?")
    t.add_argument("--data", required=True, help="??? YAML ??")
    t.add_argument("--model", default="rtdetr-l.pt", help="?????????")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=0, help="Windows ??? 0")
    t.add_argument("--project", default="runs/detect")
    t.add_argument("--name", default="train")
    t.add_argument("--device", default=0, help="GPU id ? 'cpu'")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr0", type=float, default=2e-4, help="?????")
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("val", help="????? JSON/CSV ??")
    v.add_argument("--data", required=True)
    v.add_argument("--weights", required=True)
    v.add_argument("--imgsz", type=int, default=640)
    v.add_argument("--batch", type=int, default=16)
    v.add_argument("--device", default=0)
    v.add_argument("--plots", action="store_true", help="?? PR/????")
    v.add_argument("--save-dir", default=None, help="??????")
    v.add_argument("--augment", action="store_true", help="Ultralytics 内置 TTA（非 Gamma-TTA）")
    v.set_defaults(func=cmd_val)

    i = sub.add_parser("infer", help="????????? Gamma-TTA?")
    i.add_argument("--weights", required=True)
    i.add_argument("--source", required=True, help="?????")
    i.add_argument("--save-dir", default="./outputs/viz")
    i.add_argument("--imgsz", type=int, default=640)
    i.add_argument("--device", default=0)
    i.add_argument("--conf", type=float, default=0.25)
    i.add_argument("--iou", type=float, default=0.5)
    i.add_argument("--tta", action="store_true", help="?? Gamma-TTA")
    i.add_argument(
        "--gamma-values",
        default="0.8,1.0,1.2",
        help="Gamma ????????TTA ????",
    )
    i.add_argument(
        "--tta-iou", type=float, default=0.6, help="TTA ??? NMS IoU ??"
    )
    i.set_defaults(func=cmd_infer)

    pipe = sub.add_parser("pipeline", help="?? YAML/UCCP/??/??/??")
    pipe.add_argument(
        "--root",
        default=default_root,
        help=f"??????{default_root}?",
    )
    pipe.add_argument(
        "--data",
        default=None,
        help="YAML ??? (???????????/underwater_urpc21_win.yaml)",
    )
    pipe.add_argument("--write-yaml", action="store_true", help="????? YAML")
    pipe.add_argument("--urpc21", action="store_true", help="写 YAML 时使用 URPC2021 类名")
    pipe.add_argument("--preprocess", action="store_true", help="在流水线中执行 UCCP 预处理")
    pipe.add_argument(
        "--use-uccp",
        action="store_true",
        help="训练/推理阶段改用 *_uccp 图像（若执行 preprocess 会自动开启）",
    )
    pipe.add_argument(
        "--preprocess-jobs",
        type=int,
        default=0,
        help="UCCP 预处理并行进程数（0 自动，1 单进程）",
    )
    pipe.add_argument(
        "--preprocess-skip-existing",
        action="store_true",
        help="UCCP 预处理时若输出已存在则跳过",
    )
    pipe.add_argument("--model", default="rtdetr-l.pt")
    pipe.add_argument("--epochs", type=int, default=100)
    pipe.add_argument("--imgsz", type=int, default=640)
    pipe.add_argument("--batch", type=int, default=16)
    pipe.add_argument("--workers", type=int, default=0)
    pipe.add_argument("--project", default="runs/detect")
    pipe.add_argument("--name", default="train")
    pipe.add_argument("--device", default=0)
    pipe.add_argument("--seed", type=int, default=0)
    pipe.add_argument("--lr0", type=float, default=2e-4)
    pipe.add_argument(
        "--weights",
        default=None,
        help="?????? skip-train ???????",
    )
    pipe.add_argument("--skip-train", action="store_true")
    pipe.add_argument("--skip-val", action="store_true")
    pipe.add_argument("--skip-infer", action="store_true")
    pipe.add_argument("--plots", action="store_true", help="?? PR/????")
    pipe.add_argument("--val-save-dir", default=None, help="?? JSON/CSV ????")
    pipe.add_argument("--infer-source", default=None, help="???? (???? val)")
    pipe.add_argument("--infer-out", default="./outputs/pipeline", help="???????")
    pipe.add_argument("--conf", type=float, default=0.25)
    pipe.add_argument("--iou", type=float, default=0.5)
    pipe.add_argument("--tta", action="store_true", help="?? Gamma-TTA")
    pipe.add_argument(
        "--gamma-values",
        default="0.8,1.0,1.2",
        help="Gamma ????????",
    )
    pipe.add_argument("--tta-iou", type=float, default=0.6, help="TTA ??? NMS IoU ??")
    pipe.set_defaults(func=cmd_pipeline)

    return ap


def main() -> None:
    """入口：解析命令并执行对应子命令。"""
    ap = build_cli()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
