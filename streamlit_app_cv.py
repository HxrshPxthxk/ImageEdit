import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import tempfile

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
    
    def Pixels(self,img, block_size=10):
        resized = cv.resize(img, (img.shape[1] // block_size, img.shape[0] // block_size))
        self.pix = cv.resize(resized, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
        return self.pix

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



st.set_page_config(
    page_title="Image Editor",
)

st.title("A Image Processing Webapp")

with st.sidebar:
    st.title("What can we do here?")
    page = st.radio(
        "Choose what you want to do here",
        ["**Apply filters** ", "**Flip Images**", "**Blur Images**"],
        captions=['Apply filters to your photos', "Flip Images ", "Blur Images to the intensity you want"]
    )

st.image("imageedit_cover.jpg",caption="Credits to Unsplash")


def apply_filter_sec():
    editor = Editor()
    st.title("Apply Filters to Images")
    st.markdown(f"**Apply your favourite filter to your images. The filter available are listed in the form of button in the below.**")
    filter = st.radio("Part 1",['Sepia','Canny','Inverse','GrayScale','HDR','Cartoonify','Winter','Summer','Pixels', 'Sharpen'],horizontal=True,label_visibility='hidden')


    original, filtered = st.columns(2)
    img = Image.open("coffee.jpg")
    img = np.array(img)
    if filter == 'Sepia':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.sepia(img))

    if filter == "Canny":
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.canny_edge(img))

    if filter == 'Inverse':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.get_inverse(img))

    if filter == 'GrayScale':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.changeToGrayScale(img))

    if filter == 'HDR':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.get_hdr(img))

    if filter == 'Cartoonify':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.cartoonify(img))

    if filter == 'Winter':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.winter_effect(img))

    if filter == 'Summer':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.summer_effect(img))

    if filter == 'Pixels':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.Pixels(img))

    if filter == 'Sharpen':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            st.image(editor.sharpen(img))
    st.markdown("**Upload your file here in format of JPG, PNG, and JPEG**")
    file = st.file_uploader(label="FIle upload",type=['jpg', 'png','jpeg'], label_visibility="hidden")

    if file is None:
        st.info("No file is yet uploaded. Upload your file to see results")

    if file is not None:
        img_file = Image.open(file)
        img_file = np.array(img_file, dtype=np.uint8)
        original, filtered = st.columns(2)
        if filter == 'Sepia':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.sepia(img_file)
                st.image(img_c)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == "Canny":
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.canny_edge(img_file)
                st.image(img_c)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'Inverse':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.get_inverse(img_file)
                st.image(img_c)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'GrayScale':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.changeToGrayScale(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'HDR':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.get_hdr(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'Cartoonify':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.cartoonify(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'Winter':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.winter_effect(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'Summer':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.summer_effect(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'Pixels':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.Pixels(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        if filter == 'Sharpen':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.sharpen(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)


        st.markdown("**Download your image by clicking the below button**")
        with open(temp_file_path, "rb") as file:
            btn = st.download_button(
            label="Download image",
            data=file.read(),  # Read the binary data from the file
            file_name="flitered.png",
            mime="image/png"
            )      


def flip_func():
    editor = Editor()

    st.title("Flip Your Images")
    st.markdown("**Flip Your Images as per your choice**")

    flipping = st.radio("Flipper",
                        ["**Flip Horizontally**","**Flip Vertically**", "**Flip Both**"], horizontal=True, label_visibility='hidden'
    )

    original, filtered = st.columns(2)
    img = Image.open("coffee.jpg")
    img = np.array(img)
    if flipping == '**Flip Horizontally**':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            img_c = editor.flip_horiz(img)
            st.image(img_c)
    if flipping == '**Flip Vertically**':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            img_c = editor.flip_verti(img)
            st.image(img_c)
    if flipping == '**Flip Both**':
        with original:
            st.title("Original")
            st.image(img)
        with filtered:
            st.title("With filter")
            img_c = editor.flip_both(img)
            st.image(img_c)
    
    st.markdown("**Upload your file here in format of JPG, PNG, and JPEG**")
    file = st.file_uploader(label="FIle upload",type=['jpg', 'png','jpeg'], label_visibility="hidden")

    if file is None:
        st.info("No file is yet uploaded. Upload your file to see results")

    if file is not None:
        img_file = Image.open(file)
        img_file = np.array(img_file, dtype=np.uint8)
        original, filtered = st.columns(2)

        if flipping == '**Flip Horizontally**':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.flip_horiz(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)
        if flipping == '**Flip Vertically**':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.flip_verti(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)
        if flipping == '**Flip Both**':
            with original:
                st.title("Original")
                st.image(img_file)
            with filtered:
                st.title("With filter")
                img_c = editor.flip_both(img_file)
                st.image(img_c)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    Image.fromarray(img_c).save(temp_file_path)

        



        st.markdown("**Download your image by clicking the below button**")
        with open(temp_file_path, "rb") as file:
            btn = st.download_button(
            label="Download image",
            data=file.read(),  # Read the binary data from the file
            file_name="flip.png",
            mime="image/png"
            )

def blur_sec():
    editor = Editor()
    st.title("Blur Your Images")
    st.markdown("**Blur Your Images as per your choice with your Chosen Intensity**")

    i =st.slider("Choose your Intensity", min_value=1, max_value=15, step=2)
    st.markdown("**How you image will look.**")
    original, filtered = st.columns(2)
    img = Image.open("coffee.jpg")
    img = np.array(img)
    with original:
        st.title("Original")
        st.image(img)
    with filtered:
        st.title("Blurred")
        img_c = editor.bluring(img,i)
        st.image(img_c)

    st.markdown("**Upload your file here in format of JPG, PNG, and JPEG**")
    file = st.file_uploader(label="FIle upload",type=['jpg', 'png','jpeg'], label_visibility="hidden")
    if file is None:
        st.info("No file is yet uploaded. Upload your file to see results")

    if file is not None:
        img_file = Image.open(file)
        img_file = np.array(img_file, dtype=np.uint8)
        original, filtered = st.columns(2)
        with original:
            st.title("Original")
            st.image(img_file)
        with filtered:
            st.title("Blurred")
            img_c = editor.bluring(img_file,i)
            st.image(img_c)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file_path = temp_file.name
                Image.fromarray(img_c).save(temp_file_path)

        st.markdown("**Download your image by clicking the below button**")
        with open(temp_file_path, "rb") as file:
            btn = st.download_button(
            label="Download image",
            data=file.read(),  # Read the binary data from the file
            file_name="blur.png",
            mime="image/png"
            )
    

    



if page == "**Apply filters** ":
    apply_filter_sec()
if page == "**Flip Images**":
    flip_func()
if page == "**Blur Images**":
    blur_sec()



    

















# file = None
# #file = st.file_uploader("Choose a file", type=['jpg','png','jpeg'])
# #file = st.camera_input("TAke a picture")

# editor = all.Editor()
# if file is not None:
#     file = Image.open(file)
#     file = np.array(file)
#     st.image(file)
#     imggray = editor.changeToGrayScale(file)
#     st.image(imggray)

#     imgblur = editor.bluring(file)
#     st.image(imgblur)

#     img_canny = editor.canny(file)
#     st.image(img_canny)

#     #img_crop = editor.crop(file, 400, 600,550, 750)
#     #st.image(img_crop)

#     img_hori_flip = editor.flip_horiz(file)
#     img_vert_flip = editor.flip_verti(file)
#     img_both_fiip = editor.flip_both(file)

#     st.image(img_hori_flip)
#     st.image(img_vert_flip)
#     st.image(img_both_fiip)
#     st.image(editor.cartoonify(file))
#     st.image(editor.oilpainting(file))
#     img_sepia = editor.sepia(file)
#     st.image(img_sepia)
#     st.image(editor.sharpen(file))
#     st.image(editor.get_hdr(file))
#     st.image(editor.get_inverse(file))
#     st.image(editor.summer_effect(file))
#     st.image(editor.winter_effect(file))
