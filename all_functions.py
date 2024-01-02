import cv2 as cv
import numpy as np

class Editor():
    def __init__(self):
        self.a = 5

    def changeToGrayScale(self, img):
        self.img  = img
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        return self.gray
    
    def bluring(self, img, intensity):
        self.img = img
        self.blur = cv.GaussianBlur(img, (intensity,intensity), cv.BORDER_DEFAULT)

        return self.blur
    
    def canny_edge(self,img):

        self.canny = cv.Canny(img,125,174)

        return self.canny
    
    def crop(self,img, fromheight, toheight, fromwidth,toWidth,):
        self.cropped = img[fromheight:toheight,fromwidth:toWidth]

        return self.cropped

    def flip_horiz(self,img):
        self.img_horiz = cv.flip(img,1)

        return self.img_horiz
    
    def flip_verti(self,img):
        self.img_verti = cv.flip(img,0)

        return self.img_verti
    
    def flip_both(self,img):
        self.img_both = cv.flip(img,-1)

        return self.img_both
    
    def cartoonify(self,img):

        #Convert to grayscale and apply median blur to reduce image noise
        self.grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Get the edges 
        self.edges = cv.adaptiveThreshold(self.grayimg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 5)

        #Convert to a cartoon version
        self.color = cv.bilateralFilter(img, 9, 250, 250)
        self.cartoon = cv.bitwise_and(self.color, self.color, mask=self.edges)

        return self.cartoon
    
    def oilpainting(self,img):
        self.oil = cv.xphoto.oilPainting(img,5,1)

        return self.oil

    def sepia(self,img):
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.sepia_matrix = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
        self.sepia_img = cv.transform(img, self.sepia_matrix)
        self.sepia_img[np.where(self.sepia_img > 255)] = 255
        self.sepia_img = np.array(self.sepia_img, dtype=np.uint8)
        self.sepia_img = cv.cvtColor(self.sepia_img, cv.COLOR_RGB2BGR)

        return self.sepia_img
    
    def sharpen(self, img):
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        self.img_sharpen = cv.filter2D(img, -1, kernel)
        return self.img_sharpen
    
    def get_hdr(self, img):
        self.hdr = cv.detailEnhance(img, sigma_s=12, sigma_r=0.15)
        return self.hdr
    
    def get_inverse(self, img):
        self.inv = cv.bitwise_not(img)
        return self.inv
    
    def gamma_function(self,channel, gamma):
        invGamma = 1/gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
        channel = cv.LUT(channel, table)
        return channel
    
    def winter_effect(self, img):
        img[:, :, 0] = self.gamma_function(img[:, :, 0], 1.25)
        img[:, :, 2] = self.gamma_function(img[:, :, 2], 0.75)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img[:, :, 1] = self.gamma_function(img[:, :, 1], 0.8)

        return img
    
    def summer_effect(self, img):
        img[:, :, 0] = self.gamma_function(img[:, :, 0], 0.75)
        img[:, :, 2] = self.gamma_function(img[:, :, 2], 1.25)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img[:, :, 1] = self.gamma_function(img[:, :, 1], 1.2)

        return img
    















        