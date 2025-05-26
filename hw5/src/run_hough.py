import cv2 as cv
from config import cfg
from hough_lines import main as hough_lines
from hough_circles import main as hough_circles
from pathlib import Path


for img_name in cfg["lines"].keys():
    img_path = Path(img_name)
    img = cv.imread(img_name)
    for typ, kwargs in cfg["lines"][img_name].items():
        print(f"Processing {img_path.stem}_{typ}.jpg")
        out = hough_lines(img, **kwargs)
        cv.imwrite(f"{img_path.stem}_{typ}.jpg", out)

for img_name in cfg["circles"].keys():
    img_path = Path(img_name)
    img = cv.imread(img_name)
    for typ, kwargs in cfg["circles"][img_name].items():
        print(f"Processing {img_path.stem}_{typ}.jpg")
        out = hough_circles(img, **kwargs)
        cv.imwrite(f"{img_path.stem}_{typ}.jpg", out)
