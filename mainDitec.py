"""
12/10 白天 log: 
- 增加了金币的检测, 金币的值分为10, 50, 100三种, 分类对应2, 3, 4; 
- 优化僵尸攻击逻辑: 放弃通过分类来区分攻击速度，统一攻击三次
- 增加老板键，防止退不出程序
- 优化信息打印逻辑
计划更新内容：
- 更换库为一个能够后台点击的库
- 添加简单的图形化界面

12/14 晚上 log:
- 优化部分逻辑，添加更多注释
"""

from ultralytics import YOLO
import pygetwindow
import numpy as np
import time
import os
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import cv2
import sys
from utils import bossKeyboard

ROOTPATH = os.path.dirname(os.path.abspath(__file__))  # main.py所在目录
MODEL = os.path.join(
    ROOTPATH, "runs", "ZVP_5cls_yolo11s", "weights", "best.pt"
)  # 使用训练好的 best.pt 模型
WINDOWS_TITLE = "植物大战僵尸中文版"  # 窗口标题
PROGRAM_RUNNING_FLAG = bossKeyboard.bossKeyboard(["q"])  # 全局变量，控制主程序的状态
IS_DRAW = True  # 是否绘画矩形框和展示图像

# 查找可用的中文字体（Windows 常见字体）
def _get_cn_font_path():
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",   # 微软雅黑
        r"C:\\Windows\\Fonts\\simhei.ttf", # 黑体
        r"C:\\Windows\\Fonts\\simsun.ttc", # 宋体
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

CN_FONT_PATH = _get_cn_font_path()

def draw_texts_cn(img, items, color=(0, 255, 0), font_size=20):
    """在图像上批量绘制中文文本。
    items: [{"text": str, "x": int, "y": int}]
    返回绘制后的 BGR 图像
    """
    if not items:
        return img
    if CN_FONT_PATH is None:
        # 未找到中文字体，跳过中文绘制，避免出现问号
        return img
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(CN_FONT_PATH, font_size)
        # 将 BGR 转为 RGB 的颜色
        rgb_color = (int(color[2]), int(color[1]), int(color[0]))
        for item in items:
            draw.text((item["x"], item["y"]), item["text"], font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # 避免因字体加载异常影响运行
        return img


def drawDetections(locations, _window, img, label_map=None, max_draw=9999):
    """绘制检测框与标签（仅显示，不排序、不点击）"""
    for i in range(min(len(locations), max_draw)):
        # 支持 [x, y, w, h] 或 [x, y, w, h, cls_idx]
        x, y, w, h = locations[i][:4]
        cls_idx = locations[i][4] if len(locations[i]) > 4 else None
        if IS_DRAW:
            # 中心坐标转换为左上角坐标
            x, y, w, h = x - w // 2, y - h // 2, w, h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 叠加标签（中文或英文），使用 PIL 中文字体避免问号
            if label_map is not None and cls_idx is not None:
                label_text = label_map.get(cls_idx)
                if label_text:
                    img = draw_texts_cn(
                        img,
                        [{"text": label_text, "x": x, "y": max(0, y - 20)}],
                        color=(0, 255, 0),
                        font_size=20,
                    )
    return img if IS_DRAW else None


def main():
    model = YOLO(MODEL, task="detect", verbose=False)
    classes = model.names
    # 构建索引到中文的映射：优先使用模型自带英文名，替换为中文显示
    # 兼容不同训练配置的英文名
    english_to_cn = {
        # 旧配置（金币/阳光）：
        "zombie": "僵尸",
        "sun": "阳光",
        "diamond": "钻石",
        "Coin_gold_dollar": "金币",
        "Coin_silver_dollar": "银币",
        # 新配置（植物类别）：
        "Sunflower": "向日葵",
        "Pea": "豌豆",
        "Zombie": "僵尸",
        "Sunshine": "阳光",
        "Strawberry": "草莓",
    }
    # model.names 可能是 dict 或 list
    if isinstance(classes, dict):
        idx_to_name = classes
    else:
        idx_to_name = {i: name for i, name in enumerate(classes)}
    idx_to_cn = {i: english_to_cn.get(name, name) for i, name in idx_to_name.items()}
    try:
        window = pygetwindow.getWindowsWithTitle(WINDOWS_TITLE)[0]
    except IndexError:
        print("未找到窗口，请打开植物大战僵尸游戏")
        return

    # hwnd = win32gui.FindWindow(None, WINDOWS_TITLE)
    # if not hwnd:
    #     print("未找到窗口，请打开植物大战僵尸游戏")
    #     return

    print(
        f"Root path: {ROOTPATH}, Model path: {MODEL}, Classes: {classes}, Windows title: {WINDOWS_TITLE}"
    )
    # 仅检测与显示，不做点击与计数

    print("主程序正在运行，按下 'q' 键退出...")
    while PROGRAM_RUNNING_FLAG.program_running:
        bg = time.time()
        if window:
            x, y, w, h = window.left, window.top, window.width, window.height
            shot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            shot = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
            wx, wy = shot.shape[1], shot.shape[0]
            # print(f"wx: {wx}, wy: {wy}")

            # 保留全屏内容进行检测，不做遮挡，以便检测所有目标

            shotDet = cv2.resize(shot, (640, 640))

            results = model(shotDet, task="detect", conf=0.8, verbose=False)[0]

            # resultsXY格式，二维数组：[[x,y,w,h],[x,y,w,h],...]
            # resultsCLS格式，一维数组：[0,1,0,1,...]
            # 每个标签的坐标和种类索引一一对应
            resultsXYandCLS = results.boxes.xywh.cpu().numpy()  # 所有标签的坐标
            resultsCLS = results.boxes.cls.cpu().numpy()  # 所有标签的种类索引
            # 更新数组
            resultsXYandCLS = [
                [*[int(j) for j in resultsXYandCLS[i]], int(resultsCLS[i])]
                for i in range(len(resultsXYandCLS))
            ]
            # print(resultsXYandCLS)
            detections = []

            for i in range(len(resultsXYandCLS)):
                x, y, w, h = resultsXYandCLS[i][:4]

                # 分辨率转换
                x, y, w, h = x * wx // 640, y * wy // 640, w * wx // 640, h * wy // 640
                # 收集所有检测用于绘制（不区分类别和点击）
                detections.append([x, y, w, h, resultsXYandCLS[i][4]])

            # 不进行排序，直接绘制所有检测

            if IS_DRAW:
                shot = drawDetections(detections, window, shot, idx_to_cn)
                cv2.imshow("Screen", shot)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    pass

            ed = time.time()
            sys.stdout.write(
                f"\rdetect speed: {(ed-bg)*1000:.1f} ms, detections: {len(resultsXYandCLS)}"
                + " " * 10
            )
            sys.stdout.flush()

            pass

    cv2.destroyAllWindows()


# 启动键盘监听器线程
PROGRAM_RUNNING_FLAG.startListen()

main()
