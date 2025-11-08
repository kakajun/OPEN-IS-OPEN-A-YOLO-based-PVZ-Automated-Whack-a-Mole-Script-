

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
MODEL = os.path.join(ROOTPATH, "runs", "train", "weights", "best.pt")  # 模型路径
WINDOWS_TITLE = "植物大战僵尸中文版"  # 窗口标题
ZOMBIE_SIZE = (0.06, 0.1)  # 僵尸的尺寸，宽度和高度占屏幕的比例
MONEY = [10, 50, 100]  # 金币的值
PROGRAM_RUNNING_FLAG = bossKeyboard.bossKeyboard(["q"])  # 全局变量，控制主程序的状态
IS_DRAW = True  # 是否绘画矩形框和展示图像

# 每个类别的专属颜色（BGR），索引与模型 classes 对齐
# 0: 僵尸, 1: 阳光, 2: 钻石, 3: 金币, 4: 银币, 5: 豌豆, 6: 坚果, 7: 地雷, 8: 灯笼, 9: 向日葵
CLASS_COLORS = {
    0: (0, 0, 255),      # 僵尸：红色
    1: (0, 255, 255),    # 阳光：黄色
    2: (255, 255, 0),    # 钻石：青色
    3: (0, 165, 255),    # 金币：橙色
    4: (160, 160, 160),  # 银币：灰色
    5: (0, 255, 0),      # 豌豆：绿色
    6: (19, 69, 139),    # 坚果：棕色系
    7: (255, 0, 255),    # 地雷：洋红/紫色
    8: (255, 0, 0),      # 灯笼：蓝色（BGR）
    9: (255, 0, 127),    # 向日葵：粉色
}


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
        cv2.putText(img_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
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
        # 支持 [x, y, w, h], [x, y, w, h, cls_idx], [x, y, w, h, cls_idx, conf]
        x, y, w, h = locations[i][:4]
        cls_idx = locations[i][4] if len(locations[i]) > 4 else None
        conf_val = locations[i][5] if len(locations[i]) > 5 else None
        if IS_DRAW:
            # 中心坐标转换为左上角坐标
            x, y, w, h = x - w // 2, y - h // 2, w, h
            color = CLASS_COLORS.get(cls_idx, (0, 255, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # 叠加中文标签（附加识别率）
            if label_map is not None and cls_idx is not None:
                label_cn = label_map.get(cls_idx, "？？？？？？")
                if label_cn:
                    # 在名字后面追加识别率百分比（如 85%）
                    if conf_val is not None:
                        label_cn = f"{label_cn} {int(conf_val * 100)}%"
                    img = put_text_cn(
                        img,
                        label_cn,
                        (x, max(0, y - 20)),  # PIL 字体基线不同，稍微上移
                        font_cn,
                        color,
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
    # CLASSES = ["nomal_zombie", "sun", "diamond",
#            "coin_gold_dollar", "coin_silver_dollar", "wandou", "jianguo", "dilei", "den", "xiangrikui"]

    # 兼容不同训练配置的英文名
    english_to_cn = {
        # ZVP.yaml 当前配置（10类）：
        "nomal_zombie": "僵尸",
        "sun": "阳光",
        "diamond": "钻石",
        "coin_gold_dollar": "金币",
        "coin_silver_dollar": "银币",
        "wandou": "豌豆",
        "jianguo": "坚果",
        "dilei": "地雷",
        "den": "灯笼",
        "xiangrikui": "向日葵",
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

            # 按你的要求提高阈值至 0.8
            results = model(shotDet, task="detect", conf=0.9, verbose=False)[0]

            # resultsXY格式，二维数组：[[x,y,w,h],[x,y,w,h],...]
            # resultsCLS格式，一维数组：[0,1,0,1,...]
            # 每个标签的坐标和种类索引一一对应
            resultsXY = results.boxes.xywh.cpu().numpy()  # 所有标签的坐标
            resultsCLS = results.boxes.cls.cpu().numpy()  # 所有标签的种类索引
            resultsCONF = results.boxes.conf.cpu().numpy()  # 所有标签的置信度
            # 将坐标、类别、置信度合并
            resultsXYandCLS = [
                [
                    int(resultsXY[i][0]),
                    int(resultsXY[i][1]),
                    int(resultsXY[i][2]),
                    int(resultsXY[i][3]),
                    int(resultsCLS[i]),
                    float(resultsCONF[i]),
                ]
                for i in range(len(resultsXY))
            ]
            # print(resultsXYandCLS)
            # 统一收集所有类别，避免只显示部分类别导致漏标
            allDetections = []

            for i in range(len(resultsXYandCLS)):
                x, y, w, h = resultsXYandCLS[i][:4]

                # 分辨率转换
                x, y, w, h = x * wx // 640, y * wy // 640, w * wx // 640, h * wy // 640
                cls_idx = resultsXYandCLS[i][4]
                conf_val = resultsXYandCLS[i][5] if len(
                    resultsXYandCLS[i]) > 5 else None
                if cls_idx == 0:
                    # 僵尸做一次尺寸过滤，避免误检
                    if not dropFakeZombie(w, h, wx, wy):
                        allDetections.append([x, y, w, h, cls_idx, conf_val])
                else:
                    allDetections.append([x, y, w, h, cls_idx, conf_val])

                if cls_idx in [2, 3, 4]:
                    # 统计金币/钻石计数（不影响标注）
                    coins_value_idx = cls_idx - 2
                    if 0 <= coins_value_idx < len(MONEY):
                        coinCounter += MONEY[coins_value_idx]
                    if cls_idx == 2:
                        diamandCounter += 1
                    if cls_idx == 3:
                        goldCounter += 1
                    if cls_idx == 4:
                        silverCounter += 1

            # 直接标注所有检测结果（包含向日葵等所有类别）
            shot = clickIt(allDetections, window, shot,
                           idx_to_cn, font_cn=font_cn)
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
