import imutils # untuk memperkecil ukuran video
import cv2
from src.config.Path import *
from src.config.Config import *
import os

def substract(input_file, output_folder, output_csv):
    vid = cv2.VideoCapture(input_file) # nama file yang ingin diproses
    firstFrame = None # frame pertama dari video digunakan sebagai acuan
    count = 0 # jumlah frame yang diproses, untuk penamaan file
    while True:
        frame = vid.read()
        frame = frame[1]
        if frame is None:
            break
        frame = imutils.resize(frame, width=500) # ubah ukuran video
        raw = os.path.join(Path.raw ,"%d_nobox.jpg") % count
        cv2.imwrite(raw, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # konversi ke grayscale
        #put code here
        gray = cv2.GaussianBlur(gray, (21, 21), 0) # gunakan gaussian blur untuk mengurangi noise
        if firstFrame is None:
            firstFrame = gray # ambil frame pertama video untuk acuan
            continue
        frameDelta = cv2.absdiff(firstFrame, gray) # substraction antara frame pertama dengan frame saat ini
        thresh = cv2.threshold(frameDelta, Config.threshhold_sens, 255, cv2.THRESH_BINARY)[1] # lakukan thresholding untuk mendapatkan masking
        thresh = cv2.dilate(thresh, None, iterations=2) 
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) < 1000: # batasi luas area objek yang ingin di-crop
                continue
            (x, y, w, h) = cv2.boundingRect(c) # tentukan posisi cropping
            if (h/w >= 1.5):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        save = os.path.join(output_folder ,"%d_boxed.jpg") % count
        cv2.imwrite(save, frame)
        try:
            file1 = open(output_csv, "a")
            data = str(save) + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + '\n'
            file1.writelines(data)
            file1.close()
        except:
            continue
        count += 1
    vid.release()