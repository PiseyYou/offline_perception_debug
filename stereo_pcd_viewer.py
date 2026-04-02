#!/usr/bin/env python3
"""stereo_pcd_viewer.py
双目图片分析2 — 从文件夹中读取含障碍物的点云，左侧显示双目图，右侧显示点云可视化。

用法:
    python3 stereo_pcd_viewer.py

依赖: python3, tkinter(内置), numpy, matplotlib
    pip install numpy matplotlib
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import struct
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

# ── 障碍物标签集 (与 C++ colorMap / filterLabelDect 逻辑一致) ──────────────
OBSTACLE_LABELS = {
    4,    # dynamic
    5,    # static_obstacle
    6,    # wall
    7,    # vehicle
    8,    # pole
    9,    # impassable
    10,   # depression
    100,  # pole
    101,  # obst
    102,  # fixo
    103,  # car
    104,  # stat
    105,  # dyna
    106,  # chst
    107,  # pers
}

# BGR -> RGB 颜色映射 (与 C++ initColorMap 一致)
COLOR_MAP_RGB = {
    0:   (0,   0,   0),
    1:   (0,   0,   200),   # background  蓝
    2:   (100, 255, 102),   # grass       绿
    3:   (118, 89,  0),     # road        褐
    4:   (255, 255, 0),     # dynamic     黄
    5:   (255, 0,   0),     # static_obstacle 红
    6:   (255, 165, 0),     # wall
    7:   (255, 20,  147),   # vehicle
    8:   (0,   255, 255),   # pole
    9:   (245, 130, 48),    # impassable
    10:  (0,   64,  128),   # depression
    11:  (34,  139, 34),    # bush
    12:  (255, 192, 203),   # limb_bush
    13:  (138, 43,  226),   # CES_arod
    100: (255, 0,   0),
    101: (255, 0,   0),
    102: (255, 0,   0),
    103: (255, 0,   255),   # car 洋红
    104: (255, 0,   0),
    105: (255, 255, 0),
    106: (0,   255, 255),
    107: (0,   255, 0),
}

DEFAULT_COLOR = (128, 128, 128)


def label_color(label: int):
    """返回 (R,G,B) float [0,1] 颜色。"""
    c = COLOR_MAP_RGB.get(int(label), DEFAULT_COLOR)
    return c[0] / 255.0, c[1] / 255.0, c[2] / 255.0


# ── PCD 解析器 ─────────────────────────────────────────────────────────────
def parse_pcd(path: str):
    """解析 ASCII / binary PCD 文件，返回 dict with keys x,y,z,label (numpy arrays)。"""
    with open(path, 'rb') as f:
        raw = f.read()

    # 解析头部
    header_lines = []
    header_end = 0
    for i, line in enumerate(raw.split(b'\n')):
        header_lines.append(line.decode('ascii', errors='replace').strip())
        if line.strip().upper().startswith(b'DATA'):
            header_end = sum(len(l) + 1 for l in raw.split(b'\n')[:i + 1])
            break

    meta = {}
    for line in header_lines:
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        meta[parts[0].upper()] = parts[1:]

    fields = meta.get('FIELDS', [])
    sizes  = [int(s) for s in meta.get('SIZE', [])]
    types  = meta.get('TYPE', [])
    count  = [int(c) for c in meta.get('COUNT', ['1'] * len(fields))]
    num_points = int(meta.get('POINTS', ['0'])[0])
    data_type  = meta.get('DATA', ['ascii'])[0].lower()

    # 计算各字段偏移
    point_step = sum(s * c for s, c in zip(sizes, count))
    offsets = {}
    off = 0
    for fname, sz, tp, ct in zip(fields, sizes, types, count):
        offsets[fname.lower()] = (off, sz, tp, ct)
        off += sz * ct

    # 读取数据
    if data_type == 'ascii':
        data_raw = raw[header_end:].decode('ascii', errors='replace')
        rows = [r.split() for r in data_raw.strip().split('\n') if r.strip()]
        arr = np.array(rows, dtype=float)
        result = {}
        for i, fname in enumerate(fields):
            if i < arr.shape[1]:
                result[fname.lower()] = arr[:, i]
        return result
    else:  # binary
        data_bytes = raw[header_end:]
        points = []
        for i in range(num_points):
            start = i * point_step
            if start + point_step > len(data_bytes):
                break
            chunk = data_bytes[start:start + point_step]
            row = {}
            for fname in ['x', 'y', 'z', 'rgb', 'label']:
                if fname not in offsets:
                    continue
                o, sz, tp, ct = offsets[fname]
                fmt_map = {'F': {4: 'f', 8: 'd'}, 'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                           'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
                fmt_char = fmt_map.get(tp.upper(), {}).get(sz, 'f')
                row[fname] = struct.unpack_from('<' + fmt_char, chunk, o)[0]
            points.append(row)

        result = {k: np.array([p[k] for p in points if k in p]) for k in ['x', 'y', 'z', 'label', 'rgb']}
        return result


def has_obstacle(pcd_data: dict) -> bool:
    """点云中是否含有障碍物标签。"""
    labels = pcd_data.get('label')
    if labels is None or len(labels) == 0:
        return False
    unique = set(int(l) for l in labels)
    return bool(unique & OBSTACLE_LABELS)


def find_stereo_image(img_dir: str, stem: str):
    """在 img_dir 中按 stem 查找对应的双目或左目图片。"""
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        for candidate in [stem, stem.replace('_dsg', ''), stem.replace('_sub_cdt', ''),
                          stem.replace('_mul_cdt', '')]:
            path = os.path.join(img_dir, candidate + ext)
            if os.path.isfile(path):
                return path
    # 模糊匹配
    try:
        files = os.listdir(img_dir)
    except Exception:
        return None
    for f in sorted(files):
        name, fext = os.path.splitext(f)
        if fext.lower() in ('.jpg', '.jpeg', '.png', '.bmp') and stem[:10] in name:
            return os.path.join(img_dir, f)
    return None


# ── 点云可视化 (复现 stereo_xyz_rgbl_plane_final 逻辑) ──────────────────────
IMG_H, IMG_W = 480, 640


def _get_colors(labels, rgbs=None):
    """返回每个点的 label 颜色 (N,3) float [0,1]。"""
    colors = np.zeros((len(labels), 3))
    for i, lbl in enumerate(labels):
        colors[i] = label_color(int(lbl))
    return colors


def render_views(x, y, z, labels):
    """
    复现 C++ show_xyz_rgbl_plane_point_cloud_final:
      XView (x_Image_l): Z->列, Y->行 (侧视图)
      YView (y_Image_l): Z->行(翻转), X->列 (俯视图)
      ZView (z_Image_l): X->列, Y->行 (正视图)
    返回三张 (H,W,3) uint8 图像。
    """
    if len(x) == 0:
        empty = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        return empty, empty, empty

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    def safe_scale(total, vmin, vmax):
        return total / (vmax - vmin) if vmax > vmin else 1.0

    z_scale_x = safe_scale(IMG_W, z_min, z_max)  # XView
    y_scale_x = safe_scale(IMG_H, y_min, y_max)
    z_scale_y = safe_scale(IMG_H, z_min, z_max)  # YView (行, 翻转)
    x_scale_y = safe_scale(IMG_W, x_min, x_max)
    x_scale_z = safe_scale(IMG_W, x_min, x_max)  # ZView
    y_scale_z = safe_scale(IMG_H, y_min, y_max)

    # 障碍物优先绘制，label 1/2/3 先绘制
    def sort_order(lbl):
        l = int(lbl)
        if l in (1, 2, 3):
            return 0
        if l == 2:
            return 1
        return 2

    order = sorted(range(len(labels)), key=lambda i: sort_order(labels[i]))

    xv = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    yv = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    zv = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    for i in order:
        lbl = int(labels[i])
        r_val, g_val, b_val = label_color(lbl)
        bgr = (int(b_val * 255), int(g_val * 255), int(r_val * 255))
        rgb_px = np.array([int(r_val*255), int(g_val*255), int(b_val*255)], dtype=np.uint8)

        # XView: Z->col, Y->row
        col_x = int(round(z_scale_x * (z[i] - z_min)))
        row_x = int(round(y_scale_x * (y[i] - y_min)))
        if 0 <= row_x < IMG_H and 0 <= col_x < IMG_W:
            xv[row_x, col_x] = rgb_px

        # YView: Z翻转->row, X->col
        row_y = int(round(z_scale_y * (z_max - z[i])))
        col_y = int(round(x_scale_y * (x[i] - x_min)))
        if 0 <= row_y < IMG_H and 0 <= col_y < IMG_W:
            yv[row_y, col_y] = rgb_px

        # ZView: X->col, Y->row
        col_z = int(round(x_scale_z * (x[i] - x_min)))
        row_z = int(round(y_scale_z * (y[i] - y_min)))
        if 0 <= row_z < IMG_H and 0 <= col_z < IMG_W:
            zv[row_z, col_z] = rgb_px

    # 白色边框线 (YView)
    yv[:, 0] = 255
    yv[:, -1] = 255

    return xv, yv, zv


# ── 主 GUI ────────────────────────────────────────────────────────────────
class StereoAnalysisTab2(ttk.Frame):
    """双目图片分析2 Tab。"""

    def __init__(self, parent):
        super().__init__(parent)
        self._pcd_list = []       # [(pcd_path, stem), ...] 含障碍物的帧
        self._img_dir = ''        # 图片目录
        self._cur_idx = 0
        self._lock = threading.Lock()
        self._build_ui()

    def _build_ui(self):
        # ── 顶部控制条 ──────────────────────────────────────────
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Label(ctrl, text='点云文件夹:').pack(side=tk.LEFT)
        self._pcd_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self._pcd_var, width=55).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text='浏览', command=self._browse_pcd).pack(side=tk.LEFT)

        ttk.Label(ctrl, text='  图片文件夹:').pack(side=tk.LEFT)
        self._img_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self._img_var, width=55).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text='浏览', command=self._browse_img).pack(side=tk.LEFT)

        ttk.Button(ctrl, text='加载', command=self._start_load).pack(side=tk.LEFT, padx=8)

        # ── 状态栏 ──────────────────────────────────────────────
        stat = ttk.Frame(self)
        stat.pack(side=tk.TOP, fill=tk.X, padx=6)
        self._status_var = tk.StringVar(value='请输入文件夹路径后点击加载')
        ttk.Label(stat, textvariable=self._status_var, foreground='gray').pack(side=tk.LEFT)
        self._progress = ttk.Progressbar(stat, length=200, mode='determinate')
        self._progress.pack(side=tk.LEFT, padx=8)

        # ── 导航栏 ──────────────────────────────────────────────
        nav = ttk.Frame(self)
        nav.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        ttk.Button(nav, text='◀ 上一帧', command=self._prev).pack(side=tk.LEFT)
        ttk.Button(nav, text='下一帧 ▶', command=self._next).pack(side=tk.LEFT, padx=4)
        self._frame_var = tk.StringVar(value='0 / 0')
        ttk.Label(nav, textvariable=self._frame_var).pack(side=tk.LEFT, padx=8)
        self._name_var = tk.StringVar(value='')
        ttk.Label(nav, textvariable=self._name_var, foreground='#555').pack(side=tk.LEFT)

        # ── 主显示区 ────────────────────────────────────────────
        display = ttk.Frame(self)
        display.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 左侧: 双目图片
        left_frame = ttk.LabelFrame(display, text='双目图片')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._img_fig, self._img_ax = plt.subplots(1, 1, figsize=(6, 4))
        self._img_fig.subplots_adjust(0, 0, 1, 1)
        self._img_ax.axis('off')
        self._img_canvas = FigureCanvasTkAgg(self._img_fig, master=left_frame)
        self._img_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 右侧: 点云可视化
        right_frame = ttk.LabelFrame(display, text='点云可视化 (Label着色)')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 3行2列布局: 上标签着色(XView YView ZView) + 下图例
        self._pc_fig = plt.figure(figsize=(7, 5))
        gs = self._pc_fig.add_gridspec(2, 3, height_ratios=[10, 1], hspace=0.3, wspace=0.1)
        self._ax_xv = self._pc_fig.add_subplot(gs[0, 0])
        self._ax_yv = self._pc_fig.add_subplot(gs[0, 1])
        self._ax_zv = self._pc_fig.add_subplot(gs[0, 2])
        self._ax_leg = self._pc_fig.add_subplot(gs[1, :])
        for ax, title in [(self._ax_xv, '侧视图(XView)'),
                          (self._ax_yv, '俯视图(YView)'),
                          (self._ax_zv, '正视图(ZView)')]:
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        self._ax_leg.axis('off')
        self._draw_legend()
        self._pc_canvas = FigureCanvasTkAgg(self._pc_fig, master=right_frame)
        self._pc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始空白
        self._show_empty()

    def _draw_legend(self):
        patches = [
            mpatches.Patch(color=[c/255 for c in (0,0,200)],    label='background'),
            mpatches.Patch(color=[c/255 for c in (100,255,102)],label='grass'),
            mpatches.Patch(color=[c/255 for c in (118,89,0)],   label='road'),
            mpatches.Patch(color=[c/255 for c in (255,255,0)],  label='dynamic'),
            mpatches.Patch(color=[c/255 for c in (255,0,0)],    label='obstacle'),
            mpatches.Patch(color=[c/255 for c in (255,0,255)],  label='car'),
            mpatches.Patch(color=[c/255 for c in (0,255,255)],  label='chst'),
            mpatches.Patch(color=[c/255 for c in (0,255,0)],    label='person'),
        ]
        self._ax_leg.legend(handles=patches, loc='center', ncol=8,
                            fontsize=6, frameon=False)

    # ── 浏览对话框 ─────────────────────────────────────────────
    def _browse_pcd(self):
        d = filedialog.askdirectory(title='选择点云文件夹')
        if d:
            self._pcd_var.set(d)

    def _browse_img(self):
        d = filedialog.askdirectory(title='选择图片文件夹')
        if d:
            self._img_var.set(d)

    # ── 加载 ───────────────────────────────────────────────────
    def _start_load(self):
        pcd_dir = self._pcd_var.get().strip()
        if not pcd_dir or not os.path.isdir(pcd_dir):
            messagebox.showerror('错误', '点云文件夹路径无效')
            return
        self._img_dir = self._img_var.get().strip()
        self._pcd_list = []
        self._cur_idx = 0
        self._status_var.set('扫描中...')
        self._progress['value'] = 0
        threading.Thread(target=self._load_worker, args=(pcd_dir,), daemon=True).start()

    def _load_worker(self, pcd_dir):
        result = []
        try:
            all_files = sorted([
                f for f in os.listdir(pcd_dir)
                if f.lower().endswith(('.pcd', '.pcd.bin'))
            ])
        except Exception as e:
            self.after(0, lambda: self._status_var.set(f'读取目录失败: {e}'))
            return

        total = len(all_files)
        if total == 0:
            self.after(0, lambda: self._status_var.set('未找到 .pcd 文件'))
            return

        for idx, fname in enumerate(all_files):
            path = os.path.join(pcd_dir, fname)
            try:
                data = parse_pcd(path)
                if has_obstacle(data):
                    stem = fname
                    for ext in ('.pcd', '.bin'):
                        if stem.lower().endswith(ext):
                            stem = stem[:-len(ext)]
                    result.append((path, stem, data))
            except Exception:
                pass
            pct = int((idx + 1) / total * 100)
            self.after(0, lambda p=pct, i=idx: (
                self._progress.__setitem__('value', p),
                self._status_var.set(f'扫描 {i+1}/{total}...')
            ))

        with self._lock:
            self._pcd_list = result
        self.after(0, self._on_load_done)

    def _on_load_done(self):
        n = len(self._pcd_list)
        self._status_var.set(f'找到含障碍物点云: {n} 帧')
        self._progress['value'] = 100
        if n > 0:
            self._cur_idx = 0
            self._show_frame(0)
        else:
            self._show_empty()

    # ── 导航 ───────────────────────────────────────────────────
    def _prev(self):
        if not self._pcd_list:
            return
        self._cur_idx = (self._cur_idx - 1) % len(self._pcd_list)
        self._show_frame(self._cur_idx)

    def _next(self):
        if not self._pcd_list:
            return
        self._cur_idx = (self._cur_idx + 1) % len(self._pcd_list)
        self._show_frame(self._cur_idx)

    # ── 帧显示 ─────────────────────────────────────────────────
    def _show_empty(self):
        self._img_ax.cla()
        self._img_ax.axis('off')
        self._img_ax.text(0.5, 0.5, '无图片', ha='center', va='center',
                          transform=self._img_ax.transAxes, color='gray')
        self._img_canvas.draw()

        for ax in (self._ax_xv, self._ax_yv, self._ax_zv):
            ax.cla()
            ax.axis('off')
        self._pc_canvas.draw()
        self._frame_var.set('0 / 0')
        self._name_var.set('')

    def _show_frame(self, idx):
        with self._lock:
            if not self._pcd_list or idx >= len(self._pcd_list):
                return
            pcd_path, stem, data = self._pcd_list[idx]

        n = len(self._pcd_list)
        self._frame_var.set(f'{idx + 1} / {n}')
        self._name_var.set(stem)

        # 左侧: 双目图片
        img_path = None
        if self._img_dir and os.path.isdir(self._img_dir):
            img_path = find_stereo_image(self._img_dir, stem)

        self._img_ax.cla()
        self._img_ax.axis('off')
        if img_path and os.path.isfile(img_path):
            img = plt.imread(img_path)
            self._img_ax.imshow(img)
            self._img_ax.set_title(os.path.basename(img_path), fontsize=7)
        else:
            self._img_ax.text(0.5, 0.5, f'未找到图片\n{stem}', ha='center',
                              va='center', transform=self._img_ax.transAxes, color='gray')
        self._img_canvas.draw()

        # 右侧: 点云可视化 (后台线程渲染)
        threading.Thread(target=self._render_pc, args=(data, stem), daemon=True).start()

    def _render_pc(self, data, stem):
        try:
            x = np.array(data.get('x', []), dtype=np.float32)
            y = np.array(data.get('y', []), dtype=np.float32)
            z = np.array(data.get('z', []), dtype=np.float32)
            labels = np.array(data.get('label', np.zeros(len(x))), dtype=np.int32)

            if len(x) == 0:
                self.after(0, lambda: self._status_var.set(f'{stem}: 点云为空'))
                return

            xv, yv, zv = render_views(x, y, z, labels)
        except Exception as e:
            self.after(0, lambda: self._status_var.set(f'渲染错误: {e}'))
            return

        self.after(0, lambda: self._update_pc_display(xv, yv, zv, stem, len(x)))

    def _update_pc_display(self, xv, yv, zv, stem, n_pts):
        for ax, img, title in [
            (self._ax_xv, xv, '侧视图'),
            (self._ax_yv, yv, '俯视图'),
            (self._ax_zv, zv, '正视图'),
        ]:
            ax.cla()
            ax.imshow(img, aspect='auto')
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        self._status_var.set(f'{stem}  共 {n_pts} 点')
        self._pc_canvas.draw()


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('离线感知调试 - 可视化工具')
        self.geometry('1600x900')

        # 设置中文字体
        try:
            matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei',
                                                       'Noto Sans CJK SC', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 双目图片分析2 tab
        tab2 = StereoAnalysisTab2(notebook)
        notebook.add(tab2, text='双目图片分析2')


if __name__ == '__main__':
    app = MainApp()
    app.mainloop()
