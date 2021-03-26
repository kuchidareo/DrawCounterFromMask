import numpy as np
import cv2
import os,sys
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import threading
import pathlib
import itertools



def analyze():
    def finished():
        q.set("")
        button2.config(state="active")
        messagebox.showinfo("ThresholdingImage", "終了しました")
    q.set("Thresholding...")
    i=0
    n=0

    previous_path=original_paths[0].replace(os.sep,"/")
    previous_path=previous_path[:previous_path.rfind("/")]


    cross_section_paths=[]
    filepath1 = previous_path + "/cross_section"
    path = pathlib.Path(filepath1)
    for file_or_dir in path.iterdir():
        cross_section_paths.append(str(file_or_dir))

    ribeye_mask_paths=[]
    filepath2 = previous_path + "/ribeye_mask"
    path = pathlib.Path(filepath2)
    for file_or_dir in path.iterdir():
        ribeye_mask_paths.append(str(file_or_dir))

    for cross_section_path in cross_section_paths:
        cross_section_name = cross_section_path[cross_section_path.rfind(os.sep) + 1:]
        cross_section_name = cross_section_name[:cross_section_name.rfind(".")]
        for ribeye_mask_path in ribeye_mask_paths:
            ribeye_mask_name = ribeye_mask_path[ribeye_mask_path.rfind(os.sep) + 1:]
            ribeye_mask_name = ribeye_mask_name[:ribeye_mask_name.rfind(".")]
            if cross_section_name == ribeye_mask_name:
                cross_section_img_255= cv2.imread(cross_section_path,1)
                cross_section_img = cross_section_img_255 / 255
                ribeye_mask_img = cv2.imread(ribeye_mask_path,1)
                ribeye_mask_img_gray = cv2.cvtColor(ribeye_mask_img, cv2.COLOR_BGR2GRAY)

                image_height = ribeye_mask_img_gray.shape[0]
                image_width = ribeye_mask_img_gray.shape[1]

                for x in range(image_height):
                    ribeye_mask_img_gray[x,0] = 0
                    ribeye_mask_img_gray[x,image_width-1] = 0
                for y in range(image_width):
                    ribeye_mask_img_gray[0,y] = 0
                    ribeye_mask_img_gray[image_height-1,y] = 0

                #Sobelフィルタでx方向のエッジ検出
                gray_sobelx = cv2.Sobel(ribeye_mask_img_gray,cv2.CV_32F,1,0)            
                #Sobelフィルタでy方向のエッジ検出
                gray_sobely = cv2.Sobel(ribeye_mask_img_gray,cv2.CV_32F,0,1)
                #8ビット符号なし整数変換
                gray_abs_sobelx = cv2.convertScaleAbs(gray_sobelx) 
                gray_abs_sobely = cv2.convertScaleAbs(gray_sobely)
                #重み付き和
                ribeye_sobel_edge = cv2.addWeighted(gray_abs_sobelx,0.5,gray_abs_sobely,0.5,0)

                ribeye_sobel_edge[ribeye_sobel_edge >= 100] = 255
                ribeye_sobel_edge_green = np.zeros(cross_section_img.shape)
                ribeye_sobel_edge_green[ribeye_sobel_edge >=254] = [0,255,0]

                ribeye_mask_img = ribeye_mask_img / 255
                cross_section_img_ribeye = cross_section_img * ribeye_mask_img
                cross_section_img_ribeye = cross_section_img_ribeye * 255
                g = cross_section_img_ribeye[:,:,1]
                ribeye_img_green = g[g != 0]
                q10 = np.percentile(ribeye_img_green , 10)
                thresholding_value = 0.38 * (255 - q10) + q10

                green_img = np.zeros(cross_section_img.shape)
                green_img[:,:,0] = g
                green_img[:,:,1] = g
                green_img[:,:,2] = g

                result_img = green_img.copy()
                result_img[result_img < thresholding_value] = 0
                result_img[result_img >= thresholding_value] = 255
                kernel = np.ones((2,2),np.uint8)
                result_img = cv2.dilate(result_img,kernel,iterations = 1)
                result_img = (result_img / 255) * ribeye_mask_img * 255

                result_img_rin = np.zeros(cross_section_img.shape)
                result_img_rin = result_img + ribeye_sobel_edge_green
                cross_section_img_rin = cross_section_img_255 + ribeye_sobel_edge_green

                result_img_rin[ribeye_sobel_edge_green[:,:,1] >= 254] = [0,255,0]
                cross_section_img_rin[ribeye_sobel_edge_green[:,:,1] >= 254] = [0,255,0]

                #path設定
                original_path=cross_section_path.replace(os.sep,"/")
                previous_path=original_path[:original_path.rfind("/")]
                previous_path=previous_path[:previous_path.rfind("/")]
                image_name=original_path[original_path.rfind("/")+1:]
                image_name=image_name[:image_name.rfind(".")]
                if not os.path.exists(previous_path+"/binary"):
                    os.mkdir(previous_path+"/binary")
                cv2.imwrite(previous_path+"/binary/"+image_name+"_binary.png", result_img)
                if not os.path.exists(previous_path+"/binary_rin"):
                    os.mkdir(previous_path+"/binary_rin")
                cv2.imwrite(previous_path+"/binary_rin/"+image_name+"_binary_rin.png", result_img_rin)
                if not os.path.exists(previous_path+"/cross_section_rin"):
                    os.mkdir(previous_path+"/cross_section_rin")
                cv2.imwrite(previous_path+"/cross_section_rin/"+image_name+"_cross_section_rin.png", cross_section_img_rin)
                i+=1
                q.set(str(i)+"/"+str(len(cross_section_paths)))
                n += 1
                del cross_section_img,ribeye_mask_img,g,ribeye_img_green,green_img,result_img
                break
        continue
    finished()

def callback():
    button2.config(state="disable")
    th = threading.Thread(target=analyze)
    th.start()

def sansyou1_clicked():
    iDir = os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop"
    global filepath1
    global original_paths
    original_paths=[]
    filepath1 = filedialog.askdirectory(initialdir = iDir)
    path = pathlib.Path(filepath1)
    for file_or_dir in path.iterdir():
        original_paths.append(str(file_or_dir))

    file1.set(filepath1)

if __name__ == '__main__':
    # rootの作成
    root = Tk()
    root.title('ThresholdImage')
    root.resizable(False, False)

    # Frame1
    frame1 = ttk.Frame(root, padding=20)
    frame1.grid(row=0)
    s1 = StringVar()
    s1.set('MIJ DATA FOLDER')
    label1 = ttk.Label(frame1, textvariable=s1)
    label1.grid(row=0, column=0,sticky=W)

    file1 = StringVar()
    file1_entry = ttk.Entry(frame1, textvariable=file1, width=50)
    file1_entry.grid(row=0, column=1,padx=20)

    button1 = ttk.Button(frame1, text=u'OPEN', command=sansyou1_clicked)
    button1.grid(row=0, column=2)

   

    # Frame3 startボタン
    frame3 = ttk.Frame(root, padding=(0,0,0,10))
    frame3.grid(row=1)
    button2 = ttk.Button(frame3, text='Start', command=callback)
    button2.pack()

    global q
    q= StringVar()
    q.set("")
    progress_entry = ttk.Entry(frame3, textvariable=q, width=20)
    progress_entry.pack(pady=10)
    
    root.mainloop()