import numpy as np
import cv2 as cv
def main():
    img=cv.imread('src/bouquet.png')
    cv.imshow('bouquet',img[:,:,(0,0,0)])
    cv.waitKey(0)
    image=crop(img,10,-200,100,110)
    cv.imshow('image',image)
    cv.waitKey(0)

def crop(img,x,y,w,h):
    image=img[y:y+h,x:x+w]
    return image

if __name__ == '__main__':
    main()