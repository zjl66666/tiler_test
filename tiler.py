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

# number of colors per image
COLOR_DEPTH = conf.COLOR_DEPTH
# tiles scales
RESIZING_SCALES = conf.RESIZING_SCALES
# number of pixels shifted to create each box (x,y)
PIXEL_SHIFT = conf.PIXEL_SHIFT
# multiprocessing pool size
POOL_SIZE = conf.POOL_SIZE
# if tiles can overlap
OVERLAP_TILES = conf.OVERLAP_TILES


# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255


# returns an image given its path
# 改动位置
def read_upload_img(upload_img):
    img_bytes = upload_img.getvalue()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    return img.astype('uint8')


# 改动位置
def read_tile_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    return img.astype('uint8')


# scales an image
def resize_image(img, ratio):
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img


# the most frequent color in an image and its relative frequency
def mode_color(img, ignore_alpha=False):
    counter = defaultdict(int)
    total = 0
    for y in img:
        for x in y:
            if len(x) < 4 or ignore_alpha or x[3] != 0:
                counter[tuple(x[:3])] += 1
            else:
                counter[(-1, -1, -1)] += 1
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
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)


# load and process the tiles
@st.cache_data
def load_tiles(choose_style):
    print('Loading tiles')
    tiles = defaultdict(list)

    # 这里使用预先下载好的pickle
    with open(f'./pickle/{choose_style}_tiles.pickle', 'rb') as f:
        tiles = pickle.load(f)

    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
def image_boxes(img, res):
    if not PIXEL_SHIFT:
        shift = np.flip(res)
    else:
        shift = PIXEL_SHIFT

    boxes = []
    for y in range(0, img.shape[0], shift[1]):
        for x in range(0, img.shape[1], shift[0]):
            boxes.append({
                'img': img[y:y + res[0], x:x + res[1]],
                'pos': (x, y)
            })

    return boxes


# euclidean distance between two colors
def color_distance(c1, c2):
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    return math.sqrt((c1_int[0] - c2_int[0]) ** 2 + (c1_int[1] - c2_int[1]) ** 2 + (c1_int[2] - c2_int[2]) ** 2)


# returns the most similar tile to a box (in terms of color)
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
@st.cache_data
def get_processed_image_boxes(upload_img, tiles):
    print('Getting and processing boxes')
    img = read_upload_img(upload_img)
    pool = Pool(POOL_SIZE)
    all_boxes = []

    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        boxes = image_boxes(img, res)
        modes = pool.map(mode_color, [x['img'] for x in boxes])
        most_similar_tiles = pool.starmap(most_similar_tile, zip(modes, [ts for x in range(len(modes))]))

        i = 0
        for min_dist, tile in most_similar_tiles:
            boxes[i]['min_dist'] = min_dist
            boxes[i]['tile'] = tile
            i += 1

        all_boxes += boxes

    return all_boxes, img.shape


# places a tile in the image
def place_tile(img, box):
    p1 = np.flip(box['pos'])
    p2 = p1 + box['img'].shape[:2]
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    mask = box['tile'][:, :, 3] != 0
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    if OVERLAP_TILES or not np.any(img_box[mask]):
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]


# tiles the image
@st.cache_data
def create_tiled_image(boxes, res, render=False):
    print('Creating tiled image')
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)

    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=OVERLAP_TILES)):
        place_tile(img, box)
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


# main
def main():
    st.set_page_config(page_title="马赛克图片" + emoji.emojize(':rainbow:'))
    st.balloons()
    upload_img = st.file_uploader('选择需要加工的图片' + emoji.emojize(':camera:'))
    choose_style = st.selectbox('Select the style you want to process:penguin:', tiles_map.keys())
    if upload_img:
        st.image(upload_img)
        if choose_style:
            tiles = load_tiles(choose_style)
            st.write('It may takes a few minutes⏳')
            boxes, original_res = get_processed_image_boxes(upload_img, tiles)
            img = create_tiled_image(boxes, original_res, render=conf.RENDER)
            st.image(img, caption=f'{choose_style} style output' + emoji.emojize(':lollipop:'), channels='BGR')


if __name__ == "__main__":
    main()

