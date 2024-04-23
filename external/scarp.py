
import math
import imutils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon


class hiriseJPEG:

    def __init__(self, img):
        self.IMG = img.copy()
        self.dimx = img.shape[1]
        self.dimy = img.shape[0]
        self.centroid = (int(self.dimx/2), int(self.dimy/2))

    def getMask(self, plot = False):
        arr = np.mean(self.IMG [:,0:3],axis=1)
        y_left = arr.argmax()

        arr = np.mean(self.IMG [:,-3:],axis=1)
        y_right = arr.argmax()

        arr = np.mean(self.IMG [0:3,:],axis=0)
        x_bottom = arr.argmax()

        arr = np.mean(self.IMG [-3:,:],axis=0)
        x_top = arr.argmax()

        pgon = Polygon([(0,y_left), (x_top,self.dimy), (self.dimx,y_right), (x_bottom,0)])
        mask = np.zeros(self.IMG.shape)
        points = [[x, y] for x, y in zip(*pgon.boundary.coords.xy)]
        mask = cv.fillPoly(mask, np.array([points]).astype(np.int32), color=1) # look into int32
        mask = mask.astype(bool)
        
        if plot:
            plt.imshow(self.IMG, origin = 'lower', cmap = 'gray')
            plt.imshow(mask, origin = 'lower', cmap = 'jet', alpha=0.3)

        return mask
    
    def getContour(self, obj):
        cnt = cv.findContours(obj.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        id = np.argmax([len(arr) for arr in cnt[0]])
        cnt = cnt[0][id][:,0,:]
        
        return cnt
    
    def getMinBox(self, cnt, plot = False):
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.intp(np.append(box, [box[0]], axis=0))
        self.offset_angle = np.radians(rect[-1])

        if plot:
            plt.imshow(self.IMG, origin = 'lower', cmap = 'gray')
            plt.plot(cnt[:,0], cnt[:,1], 'ro', markersize=0.5)
            plt.plot(box[:,0], box[:,1], 'b--', linewidth=1)
            self.showAngle(rect[0], np.radians(rect[-1]))
        
        return {'center' : rect[0][1::-1], 'angle' : rect[-1], 'dims': (int(math.dist(box[1], box[2])), int(math.dist(box[0], box[1])))}
    
    def getExtremes(self, cnt, rot_inv = None, plot = False):
        left = tuple(cnt[cnt[:, 0].argmin()])
        
        right = tuple(cnt[cnt[:, 0].argmax()])
        top = tuple(cnt[cnt[:, 1].argmin()])
        bottom = tuple(cnt[cnt[:, 1].argmax()])
        extremes = np.vstack([left, right, top, bottom])
        if rot_inv:
            ones = np.ones(shape=(len(extremes), 1))
            extremes = np.hstack([extremes, ones])
            extremes = np.rint(rot_inv.dot(extremes.T).T)

        if plot:
            plt.imshow(self.IMG, origin = 'lower', cmap = 'gray')
            plt.plot(cnt[:,0], cnt[:,1], 'bo', markersize=0.5)
            plt.plot(extremes[:,0], extremes[:,1], 'ro', markersize=5)

        return extremes
    
    def cropToMask(self, mask, minbox, margin = 100, plot = False):
        rot_img = imutils.rotate_bound(self.IMG, -minbox['angle'])
        rot_mask = imutils.rotate_bound(mask.astype(np.uint8), -minbox['angle'])

        rot_mask_cnt = self.getContour(rot_mask)
        extremes = self.getExtremes(rot_mask_cnt)

        x1 = extremes[:,0].min() + margin
        x2 = extremes[:,0].max() - margin
        y1 = extremes[:,1].min() + margin
        y2 = extremes[:,1].max() - margin

        img_cropped = rot_img[y1:y2,x1:x2]

        if plot:
            plt.imshow(img_cropped, origin = 'lower', cmap = 'gray')

        return img_cropped
