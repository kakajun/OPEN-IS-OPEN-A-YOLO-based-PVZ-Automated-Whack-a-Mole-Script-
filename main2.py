

from ultralytics import YOLO
import pygetwindow
import numpy as np
import time
import os
from PIL import ImageGrab
from PIL import ImageFont, ImageDraw
from PIL import Image
import cv2
import sys
from utils import bossKeyboard

ROOTPATH = os.path.dirname(os.path.abspath(__file__))  # main.py所在目录
MODEL = os.path.join(ROOTPATH, "models", "best copy.pt")  # 模型路径
WINDOWS_TITLE = "植物大战僵尸中文版"  # 窗口标题
ZOMBIE_SIZE = (0.06, 0.1)  # 僵尸的尺寸，宽度和高度占屏幕的比例
MONEY = [10, 50, 100]  # 金币的值
PROGRAM_RUNNING_FLAG = bossKeyboard.bossKeyboard(["q"])  # 全局变量，控制主程序的状态
IS_DRAW = True  # 是否绘画矩形框和展示图像


def load_cn_font(font_size=20):
    """尝试加载系统中文字体，避免中文显示为问号"""
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",      # 微软雅黑
        r"C:\\Windows\\Fonts\\SimHei.ttf",    # 黑体
        r"C:\\Windows\\Fonts\\SimSun.ttf",    # 宋体
        r"C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc",  # 思源黑体（若安装）
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return None


def put_text_cn(img_bgr, text, xy, font, color=(0, 255, 0)):
    """在 OpenCV 图像上用中文字体绘制文本"""
    if font is None:
        # 没有中文字体时退化为英文/问号显示
        cv2.putText(img_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img_bgr
    # 将 BGR 转为 RGB，使用 PIL 绘制中文，再转回 BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


def dropFakeZombie(_x, _y, _wx, _wy) -> bool:
    """判断是否是识别错误的僵尸"""
    return _x < _wx * ZOMBIE_SIZE[0] or _y < _wy * ZOMBIE_SIZE[1]


def clickIt(locations, _window, img, label_map=None, clickTimes=1, clickNum=2, font_cn=None):
    """在指定位置绘制标注（不进行任何点击操作，全部标注）"""
    for i in range(len(locations)):
        # 支持 [x, y, w, h] 或 [x, y, w, h, cls_idx]
        x, y, w, h = locations[i][:4]
        cls_idx = locations[i][4] if len(locations[i]) > 4 else None
        if IS_DRAW:
            # 中心坐标转换为左上角坐标
            x, y, w, h = x - w // 2, y - h // 2, w, h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 叠加中文标签
            if label_map is not None and cls_idx is not None:
                label_cn = label_map.get(cls_idx, "？？？？？？")
                if label_cn:
                    img = put_text_cn(
                        img,
                        label_cn,
                        (x, max(0, y - 20)),  # PIL 字体基线不同，稍微上移
                        font_cn,
                        (0, 255, 0),
                    )
    if IS_DRAW:
        return img
    return


def main():
    model = YOLO(MODEL, task="detect", verbose=False)
    classes = model.names
    # 加载中文字体，避免 cv2.putText 将中文显示为问号
    font_cn = load_cn_font(22)
    # 构建索引到中文的映射：优先使用模型自带英文名，替换为中文显示
    # 兼容不同训练配置的英文名
    english_to_cn = {
        # ZVP.yaml 当前配置（5类）：
        "nomal_zombie": "僵尸",
        "sun": "阳光",
        "diamond": "钻石",
        "qcoin_gold_dollar": "金币",
        "coin_silver_dollar": "银币",
        # 兼容旧命名/大小写变体：
        "zombie": "僵尸",
        "Zombie": "僵尸",
        "Sun": "阳光",
        "Sunshine": "阳光",
        "Diamond": "钻石",
        "Coin_gold_dollar": "金币",
        "Coin_silver_dollar": "银币",
    }
    # model.names 可能是 dict 或 list
    if isinstance(classes, dict):
        idx_to_name = classes
    else:
        idx_to_name = {i: name for i, name in enumerate(classes)}
    idx_to_cn = {i: english_to_cn.get(name, name)
                 for i, name in idx_to_name.items()}
    try:
        window = pygetwindow.getWindowsWithTitle(WINDOWS_TITLE)[0]
    except IndexError:
        print("未找到窗口，请打开植物大战僵尸游戏")
        return

    print(
        f"Root path: {ROOTPATH}, Model path: {MODEL}, Classes: {classes}, Windows title: {WINDOWS_TITLE}"
    )
    coinCounter = 0
    diamandCounter = 0
    goldCounter = 0
    silverCounter = 0

    print("主程序正在运行，按下 'q' 键退出...")
    while PROGRAM_RUNNING_FLAG.program_running:
        bg = time.time()
        if window:
            x, y, w, h = window.left, window.top, window.width, window.height
            shot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            shot = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
            wx, wy = shot.shape[1], shot.shape[0]
            # print(f"wx: {wx}, wy: {wy}")

            # 遮挡不需要的区域，在指定区域绘画矩形
            shot = cv2.rectangle(
                shot, (0, 0), (int(wx * 0.7), int(wy * 0.17)), (0, 0, 0), -1
            )
            shot = cv2.rectangle(shot, (0, int(wy * 0.95)),
                                 (wx, wy), (0, 0, 0), -1)
            shot = cv2.rectangle(
                shot, (0, int(wy * 0.85)), (int(wx * 0.1), wy), (0, 0, 0), -1
            )

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
            zombies = []
            coins = []
            suns = []

            for i in range(len(resultsXYandCLS)):
                x, y, w, h = resultsXYandCLS[i][:4]

                # 分辨率转换
                x, y, w, h = x * wx // 640, y * wy // 640, w * wx // 640, h * wy // 640
                if resultsXYandCLS[i][4] == 0:
                    # print(f"Zombie: {w/wx} and {h/wy}")
                    if not dropFakeZombie(w, h, wx, wy):
                        zombies.append([x, y, w, h, resultsXYandCLS[i][4]])

                if resultsXYandCLS[i][4] == 1:
                    suns.append([x, y, w, h, resultsXYandCLS[i][4]])

                if resultsXYandCLS[i][4] in [2, 3, 4]:
                    coins.append([x, y, w, h, resultsXYandCLS[i][4]])
                    coinCounter += MONEY[resultsXYandCLS[i][4] - 2]
                    if resultsXYandCLS[i][4] == 2:
                        diamandCounter += 1
                    if resultsXYandCLS[i][4] == 3:
                        goldCounter += 1
                    if resultsXYandCLS[i][4] == 4:
                        silverCounter += 1

            # 直接标注所有检测结果，不进行排序
            shot = clickIt(suns, window, shot, idx_to_cn, font_cn=font_cn)
            shot = clickIt(coins, window, shot, idx_to_cn, font_cn=font_cn)
            shot = clickIt(zombies, window, shot, idx_to_cn, font_cn=font_cn)
            cv2.imshow("Screen", shot)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                pass

            ed = time.time()
            sys.stdout.write(
                f"\rdetect speed: {ed-bg} ms, moneyFound: {coinCounter}, diamand_count: {diamandCounter}, gold_count: {goldCounter}, silver_count: {silverCounter}"
                + " " * 10
            )
            sys.stdout.flush()

            pass

    cv2.destroyAllWindows()


# 启动键盘监听器线程
PROGRAM_RUNNING_FLAG.startListen()

main()
