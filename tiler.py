import cv2
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import math
import pickle
import conf
from time import sleep
import pathlib
import emoji
import streamlit as st

# 每张图像的颜色数
COLOR_DEPTH = conf.COLOR_DEPTH
# tiles scales 瓷砖比例
RESIZING_SCALES = conf.RESIZING_SCALES
# number of pixels shifted to create each box (x,y) 创建每个框的像素数 （x，y）
PIXEL_SHIFT = conf.PIXEL_SHIFT
# multiprocessing pool size 多处理池大小
POOL_SIZE = conf.POOL_SIZE
# if tiles can overlap 如果磁贴可以重叠
OVERLAP_TILES = conf.OVERLAP_TILES


# reduces the number of colors in an image
def color_quantization(img, n_colors):
    # np.round作用是四舍五入
    return np.round(img / 255 * n_colors) / n_colors * 255


# returns an image given its path
# 改动位置
def read_upload_img(upload_img):
    img_bytes = upload_img.getvalue()  # 读取上传的图片
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)  # 转换为opencv格式
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:  # 如果是3通道的图片
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    return img.astype('uint8')  # 返回uint8格式的图片，因为opencv的图片格式是uint8


# 改动位置
def read_tile_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 读取包括alpha通道的图片
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    return img.astype('uint8')


# scales an image
def resize_image(img, ratio):
    # img.shape[1]是宽，img.shape[0]是高
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img


# the most frequent color in an image and its relative frequency
# 图像中最常见的颜色及其相对频率
def mode_color(img, ignore_alpha=False):
    counter = defaultdict(int)
    total = 0
    for y in img:
        for x in y:
            if len(x) < 4 or ignore_alpha or x[3] != 0:  # 如果没有alpha通道或者alpha通道不为0
                counter[tuple(x[:3])] += 1  # 只取前三个通道
            else:
                counter[(-1, -1, -1)] += 1  # 透明像素
            total += 1

    if total > 0:
        mode_color = max(counter, key=counter.get)
        if mode_color == (-1, -1, -1):
            return None, None
        else:
            return mode_color, counter[mode_color] / total
    else:
        return None, None


# displays an image
def show_image(img, wait=True):
    cv2.imshow('img', img)
    if wait:
        cv2.waitKey(0)  # 0表示一直等待
    else:
        cv2.waitKey(1)  # 1表示等待1ms


# load and process the tiles
# 加载和处理磁贴
def load_tiles(path):
    print('Loading tiles')
    tiles = defaultdict(list)

    # 这里使用预先下载好的pickle
    name = str(path).split('\\')[-1].lstrip('gen_')
    with open(f'./pickle/{name}_tiles.pickle', 'rb') as f:
        tiles = pickle.load(f)

    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
def image_boxes(img, res):
    if not PIXEL_SHIFT:
        shift = np.flip(res)  # np.flip是将数组的维度反转,例如[1,2] -> [2,1]
    else:
        shift = PIXEL_SHIFT

    boxes = []  # 存放每个框的信息
    # shift决定了每个框的大小
    # res
    for y in range(0, img.shape[0], shift[1]):
        for x in range(0, img.shape[1], shift[0]):
            boxes.append({
                'img': img[y:y + res[0], x:x + res[1]],
                'pos': (x, y)
            })

    return boxes


# euclidean distance between two colors
# 计算两个颜色之间的欧几里得距离
def color_distance(c1, c2):
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    return math.sqrt((c1_int[0] - c2_int[0]) ** 2 + (c1_int[1] - c2_int[1]) ** 2 + (c1_int[2] - c2_int[2]) ** 2)


# returns the most similar tile to a box (in terms of color)
# 返回与框最相似的磁贴（就颜色而言）
def most_similar_tile(box_mode_freq, tiles):
    if not box_mode_freq[0]:
        return 0, np.zeros(shape=tiles[0]['tile'].shape)
    else:
        min_distance = None
        min_tile_img = None
        for t in tiles:
            dist = (1 + color_distance(box_mode_freq[0], t['mode'])) / box_mode_freq[1]
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_tile_img = t['tile']
        return (min_distance, min_tile_img)


# builds the boxes and finds the best tile for each one
def get_processed_image_boxes(upload_img, tiles):
    print('Getting and processing boxes')
    img = read_upload_img(upload_img)
    pool = Pool(POOL_SIZE)
    all_boxes = []
    # ts
    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        boxes = image_boxes(img, res)
        modes = pool.map(mode_color, [x['img'] for x in boxes])
        # pool.starmap()函数的作用和map()函数类似，区别在于map()函数会将参数按照顺序传递给函数，而starmap()函数会将参数解包后传递给函数
        most_similar_tiles = pool.starmap(most_similar_tile, zip(modes, [ts for x in range(len(modes))]))

        i = 0
        for min_dist, tile in most_similar_tiles:
            boxes[i]['min_dist'] = min_dist
            boxes[i]['tile'] = tile
            i += 1

        all_boxes += boxes
    # all_boxes是一个列表，每个元素是一个字典，字典中包含了每个框的信息,信息包括框的位置，框的图片，框的最小距离，框的磁贴
    return all_boxes, img.shape


# places a tile in the image 在图像中放置一个瓷砖
def place_tile(img, box):
    p1 = np.flip(box['pos'])
    p2 = p1 + box['img'].shape[:2]
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    mask = box['tile'][:, :, 3] != 0
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    if OVERLAP_TILES or not np.any(img_box[mask]):
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]


# tiles the image平铺图像
def create_tiled_image(boxes, res, render=False):
    print('Creating tiled image')
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=OVERLAP_TILES)):
        place_tile(img, box)  # 在图像中放置一个瓷砖
        if render:
            show_image(img, wait=False)
            sleep(0.025)

    return img


# 改动位置
def find_path(tilesname):
    start = pathlib.Path(r'./tiles')
    for i in start.iterdir():
        for j in i.iterdir():
            if j.is_dir() and tilesname in str(j):
                return j


# 提前通过find_path类似的方法得到下面的字典
tiles_map = {'at': 'gen_at', 'circle_100': 'gen_circle_100', 'circle_200': 'gen_circle_200', 'clip': 'gen_clip',
             'heart': 'gen_heart', 'lego_h': 'gen_lego_h', 'lego_v': 'gen_lego_v', 'line_h': 'gen_line_h',
             'line_v': 'gen_line_v', 'minecraft': 'gen_mini', 'plus': 'gen_plus', 'times': 'gen_times',
             'wave': 'gen_wave'}


def example_tile():
    start = pathlib.Path(r'./tiles_example')
    c1, c2, c3, c4, c5 = st.columns(5)
    c6, c7, c8, c9, c10 = st.columns(5)
    i = 1
    for tiler in start.iterdir():
        eval(f'c{i}').image(r'./tiles_example/' + str(tiler.name), caption=str(tiler.name.split('.')[0]))
        i += 1
info1 = """Tiler 是一种使用各种其他较小图像（图块）创建图像的工具。它与其他马赛克工具不同，因为它可以适应多种形状和尺寸的瓷砖（即不限于正方形）。"""
info2 = """图像可以用圆形、线条、波浪、十字绣、乐高积木、我的世界积木、回形针、字母......构建出无限的可能性！"""
# main
def main():
    st.set_page_config(page_title="马赛克图片" + emoji.emojize(':rainbow:'))
    st.title('马赛克图片' + emoji.emojize(':rainbow:'))
    st.subheader(info1)
    st.subheader(info2)
    st.balloons()
    upload_img = st.file_uploader('选择需要加工的图片' + emoji.emojize(':camera:'))
    choose_style = st.selectbox('Select the style you want to process:penguin:', tiles_map.keys())
    st.sidebar.image('./images_example/starry_night_circles_25x25.png', caption='circle星空')
    st.sidebar.image('./images_example/cake_circles.png', caption='cicle蛋糕')
    st.sidebar.image('./images_example/github_logo_at.png', caption='atGithub标志')
    if upload_img:
        st.image(upload_img)
        if choose_style:
            tiles = load_tiles(find_path(choose_style))
            st.write('It may takes a few minutes⏳请耐心等待')
            boxes, original_res = get_processed_image_boxes(upload_img, tiles)
            img = create_tiled_image(boxes, original_res, render=conf.RENDER)
            st.image(img, caption=f'{choose_style} style output' + emoji.emojize(':lollipop:'), channels='BGR')
    example_tile()


if __name__ == "__main__":
    main()
