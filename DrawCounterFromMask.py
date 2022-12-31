import numpy as np
import cv2
import csv
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

    coef_MP = float(coef_MP_txtbox.get())
    lowest_MP = float(lowest_MP_txtbox.get())/100
    highest_MP = float(highest_MP_txtbox.get())/100

    with open("Threshold_values.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ファイル名","一回目の閾値", "一回目のMP", "新閾値", "OTSU閾値(ある場合)", "最終MP", "ロース芯L10"])
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

                    binary_marbling_img = green_img.copy()
                    binary_marbling_img[binary_marbling_img < thresholding_value] = 0
                    binary_marbling_img[binary_marbling_img >= thresholding_value] = 255
                    binary_marbling_img_gray = cv2.cvtColor(binary_marbling_img.astype('float32'), cv2.COLOR_BGR2GRAY)
                    # 仮脂肪面積割合の計算
                    ribeye_area_cnt = cv2.countNonZero(ribeye_mask_img_gray)
                    marbling_area_cnt = cv2.countNonZero(binary_marbling_img_gray)
                    marbling_per = marbling_area_cnt / ribeye_area_cnt

                    # marbling_per < 0.1 で、閾値再計算。0.1 < marbling_per で大津の判別方式を利用して閾値再計算。
                    # print(marbling_per)
                    if marbling_per < highest_MP:
                        if marbling_per < lowest_MP:
                            new_thresholding_value = 0.38 * (255 - q10) + q10
                        else:
                            new_thresholding_value = 0.38 * (1 - coef_MP * (marbling_per - lowest_MP)) * (255 - q10) + (1 - coef_MP * (marbling_per - lowest_MP)) * q10
                        # print("kuchida"+str(new_thresholding_value))
                        binary_marbling_img = cv2.cvtColor(green_img.astype("uint8"), cv2.COLOR_BGR2GRAY)
                        binary_marbling_img[binary_marbling_img < new_thresholding_value] = 0
                        binary_marbling_img[binary_marbling_img >= new_thresholding_value] = 255
                        # 最終脂肪面積割合の計算
                        marbling_area_cnt = cv2.countNonZero(binary_marbling_img)
                        final_marbling_per = marbling_area_cnt / ribeye_area_cnt
                        writer.writerow([cross_section_name, thresholding_value, marbling_per, new_thresholding_value, "", final_marbling_per, q10])
                        binary_marbling_img = cv2.cvtColor(binary_marbling_img, cv2.COLOR_GRAY2BGR)
                    else:
                        new_thresholding_value_kuchida = 0.38 * (1 - coef_MP * marbling_per) * (255 - q10) + q10
                        # 大津の判別方式を適用するため、グレースケールに変更
                        binary_marbling_img = cv2.cvtColor(green_img.astype("uint8"), cv2.COLOR_BGR2GRAY)
                        new_thresholding_value, otsu_img = cv2.threshold(ribeye_img_green.astype("uint8"), 0,255, cv2.THRESH_OTSU)
                        # print("otsu"+str(new_thresholding_value))
                        binary_marbling_img[binary_marbling_img < new_thresholding_value] = 0
                        binary_marbling_img[binary_marbling_img >= new_thresholding_value] = 255
                        # 最終脂肪面積割合の計算
                        marbling_area_cnt = cv2.countNonZero(binary_marbling_img)
                        final_marbling_per = marbling_area_cnt / ribeye_area_cnt
                        writer.writerow([cross_section_name, thresholding_value, marbling_per, new_thresholding_value_kuchida, new_thresholding_value, final_marbling_per, q10])
                        binary_marbling_img = cv2.cvtColor(binary_marbling_img, cv2.COLOR_GRAY2BGR)
                    


                    binary_marbling_img_rin = np.zeros(cross_section_img.shape)
                    binary_marbling_img_rin = binary_marbling_img + ribeye_sobel_edge_green
                    cross_section_img_rin = cross_section_img_255 + ribeye_sobel_edge_green

                    binary_marbling_img_rin[ribeye_sobel_edge_green[:,:,1] >= 254] = [0,255,0]
                    cross_section_img_rin[ribeye_sobel_edge_green[:,:,1] >= 254] = [0,255,0]

                    #path設定
                    original_path=cross_section_path.replace(os.sep,"/")
                    previous_path=original_path[:original_path.rfind("/")]
                    previous_path=previous_path[:previous_path.rfind("/")]
                    image_name=original_path[original_path.rfind("/")+1:]
                    image_name=image_name[:image_name.rfind(".")]
                    if not os.path.exists(previous_path+"/binary"):
                        os.mkdir(previous_path+"/binary")
                    cv2.imwrite(previous_path+"/binary/"+image_name+"_binary.png", binary_marbling_img)
                    if not os.path.exists(previous_path+"/binary_rin"):
                        os.mkdir(previous_path+"/binary_rin")
                    cv2.imwrite(previous_path+"/binary_rin/"+image_name+"_binary_rin.png", binary_marbling_img_rin)
                    if not os.path.exists(previous_path+"/cross_section_rin"):
                        os.mkdir(previous_path+"/cross_section_rin")
                    cv2.imwrite(previous_path+"/cross_section_rin/"+image_name+"_cross_section_rin.png", cross_section_img_rin)
                    i+=1
                    q.set(str(i)+"/"+str(len(cross_section_paths)))
                    n += 1
                    del cross_section_img,ribeye_mask_img,g,ribeye_img_green,green_img,binary_marbling_img
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

    # Frame
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

    s2 = StringVar()
    s2.set('MPの係数')
    label2 = ttk.Label(frame1, textvariable=s2)
    label2.grid(row=1, column=0)
    coef_MP_txtbox = ttk.Entry(frame1)
    coef_MP_txtbox.grid(row=1, column=1)

    s3 = StringVar()
    s3.set('最低MP%')
    label3 = ttk.Label(frame1, textvariable=s3)
    label3.grid(row=2, column=0)
    lowest_MP_txtbox = ttk.Entry(frame1)
    lowest_MP_txtbox.grid(row=2, column=1)

    s4 = StringVar()
    s4.set('OTSU MP%')
    label4 = ttk.Label(frame1, textvariable=s4)
    label4.grid(row=3, column=0)
    highest_MP_txtbox = ttk.Entry(frame1)
    highest_MP_txtbox.grid(row=3, column=1)

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