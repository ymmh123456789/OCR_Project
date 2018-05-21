import cv2
import numpy as np
import statistics
from keras.models import load_model
# import os
# from time import gmtime, strftime
# import pytesseract
# from PIL import Image

model = load_model("NEW_blackB_whiteT1.0_model_2.h5")
one = 0
two = 0
three = 0
four = 0
pre_one = 0
pre_two = 0
pre_three = 0
pre_four = 0
def horizontal_pro(image):
    '''
    :param img: 輸入的圖片
    :return: 水平投影的List
    '''
    # 水平投影
    h, w = image.shape
    hist = []
    for x in range(h):
        hist.append(cv2.countNonZero(image[x, :]))

    return hist


def verical_pro(image):
    '''
    :param img: 輸入的圖片
    :return: 垂直投影的List
    '''
    # 垂直投影
    h, w = image.shape
    hist = []
    for x in range(w):
        hist.append(cv2.countNonZero(image[:, x]))

    return hist

def rotate(img, rotate_range):
    '''
    :param img: 輸入的原始圖片
    :param rotate_range: 向左向右嘗試旋轉的角度
    :return: 最好的旋轉角度
    '''
    h, w = img.shape
    count_col = []  # 紀錄每個角度所對應的空白col數
    for i in range((-rotate_range) * 10, (rotate_range + 1) * 10):
        i /= float(10)
        # 第一個參數為旋轉中心，第二個參數為旋轉角度，第三個參數為縮放比例
        M = cv2.getRotationMatrix2D((w / 2, h / 2), i, 1)
        # 第三個參數為變換後的圖像大小
        res = cv2.warpAffine(img, M, (w, h))
        # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow('test1', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('test1', 600, 800)
        # cv2.imshow('test1', res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        hist = verical_pro(res)
        count_col.append(np.std(hist))  # 利用標準差找出最好的旋轉角度
        # print(np.std(hist))
    # 找出空白col最多的那個角度
    maxiumn = 0
    best_angle = 0
    for x in range(len(count_col)):
        if count_col[x] > maxiumn:
            maxiumn = count_col[x]
            best_angle = x / float(10) - rotate_range
    return best_angle

class image:
    '''
    An Image information PDF2JPG5/DD1377QFP000AA001-1.jpg
     self.img -> original image
     self.row , self.col -> self.shape
     self.BinImg -> Binary Image after auto Rotated
    '''
    def __init__(self,filename,words):
        self.img = cv2.imread(filename,0)
        self.row , self.col = self.img.shape
        self.filename = filename.split("/")[-1]
        self.words = words

    def show(self):
        cv2.namedWindow('test1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('test1', 600, 800)
        cv2.imshow('test1', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def pre_process(self):
        '''
        find BINARY image and auto rotate return a text image without border
        http://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
        :return:
        '''
        ret, self.BinImg = cv2.threshold(self.img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # coords = np.column_stack(np.where(self.BinImg > ret))
        # angle = cv2.minAreaRect(coords)[-1]
        # if angle < -45:
        #     angle = -(90 + angle)
        # else:
        #     angle = -angle
        # print (angle)
        angle = rotate(self.BinImg, 5)
        # print (angle)
        center = (self.col/2,self.row/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.BinImg = cv2.warpAffine(self.BinImg, M, (self.col, self.row), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        coords = np.column_stack(np.where(self.BinImg > ret))
        Big = np.amax(coords, axis=0)
        Small = np.amin(coords, axis=0)
        self.BinImg = self.BinImg[Small[0]:Big[0], Small[1]:Big[1]]
        hist1 = horizontal_pro(self.BinImg)
        # cv2.namedWindow('test1', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('test1', 600, 800)
        # cv2.imshow('test1', self.BinImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        h, w = self.BinImg.shape
        find = False
        top = 0
        bottom = h
        for i in range(int(len(hist1) / 10)):
            if hist1[i] > 0.9 * w:
                find = True
            elif find:
                top = i
                find = False
        for i in range(len(hist1) - 1, (int(len(hist1) / 10) * 9), -1):
            if hist1[i] > 0.9 * w:
                find = True
            elif find:
                bottom = i
                find = False
        self.BinImg = self.BinImg[top+10:bottom-5, :]  # ???單頁上下邊界
        # cv2.namedWindow('test1', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('test1', 600, 800)
        # cv2.imshow('test1', self.BinImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return self.BinImg

    def new_Vertical(self):
        '''
        切出一行一行的字
        :return:
        '''
        row, col = self.BinImg.shape
        hist = verical_pro(self.BinImg)
        line1 = []
        line2 = []
        last_line = 0
        # left to right
        for i in range(10, len(hist) - 10):
            if hist[i] < hist[i + 5] / 2 and hist[i] < hist[i + 8] / 2 and hist[i] - min(
                    hist[20:-20]) < 10 and i > last_line + 10:
                line1.append(i)
                # self.BinImg[:, i] = 127
                last_line = i
                i += 5
        # right to left
        last_line = len(hist)
        for i in range(len(hist) - 10, 10, -1):
            if hist[i] < hist[i - 5] / 2 and hist[i] < hist[i - 8] / 2 and hist[i] - min(
                    hist[20:-20]) < 10 and i < last_line - 10:
                line2.append(i)
                # self.BinImg[:, i] = 126
                last_line = i
                i -= 5
        # cv2.imshow("all", self.BinImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 把兩個合起來然後sort
        line = line1 + line2
        line.sort()
        tmp = []
        # 找出真正線 → 兩兩距離相差30且區間白點數高於500者
        for i in range(len(line) - 1):
            if line[i + 1] - line[i] > 30 and sum(hist[line[i]:line[i + 1]]) > 500:
                tmp.append(line[i])
                tmp.append(line[i + 1])
                i += 1
        self.__segment = []
        for i in range(0, len(tmp) - 1, 2):
            # 若為最左或最右行，為避免邊線因過寬而被切割出而多做的判斷
            if i == 0 or i == len(tmp)-2:
                # print("row: " + str(row))
                tmp_bool = False
                for j in range(tmp[i], tmp[i+1]):
                    if hist[j] > 0.8*row:
                        # print(hist, 0.8*row)
                        tmp_bool = True
                    else:
                        pass
                if not tmp_bool:
                    self.__segment.append(self.BinImg[:, tmp[i]:tmp[i+1]])
            else:
                self.__segment.append(self.BinImg[:, tmp[i]:tmp[i + 1]])
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("test", 40, 900)
            # cv2.imshow('test', self.__segment[len(self.__segment)-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        LINE_THRESHOLD = 0.3
        HIST_THRESHOLD = 0.7
        # combine = False
        for index, img in enumerate(self.__segment):
            left_img = img[:, :int(img.shape[1] * LINE_THRESHOLD)]
            right_img = img[:, -int(img.shape[1] * LINE_THRESHOLD):]
            left_pro, right_pro = verical_pro(left_img), verical_pro(right_img)
            left_info, right_info = np.max(left_pro), np.max(right_pro)
            # org = self.__segment[index].copy()
            if left_info > img.shape[0] * HIST_THRESHOLD:
                self.__segment[index] = self.__segment[index][:, np.argmax(left_pro) + 3:]
                # combine = True
            if right_info > img.shape[0] * HIST_THRESHOLD:
                self.__segment[index] = self.__segment[index][:, :np.argmax(right_pro) - len(right_pro) - 1 - 3]
                # combine = True
                # if combine:
                #     if left_info > img.shape[0] * HIST_THRESHOLD and np.argmax(left_pro) > img.shape[1] * 0.1:
                #         print("(%d, %d)" % (img.shape[1], np.argmax(left_pro)))
                #         cv2.imwrite('line1/' + self.filename + str(index) + 'O.jpg', org)
                #         cv2.imwrite('line1/' + self.filename + str(index) + 'Z.jpg', self.__segment[index])
                #     if right_info > img.shape[0] * HIST_THRESHOLD and abs(np.argmax(right_pro) - len(right_pro) - 1) > img.shape[1] * 0.1:
                #         print("(%d, %d)" % (img.shape[1], abs(np.argmax(right_pro) - len(right_pro) - 1)))
                #         cv2.imwrite('line1/' + self.filename + str(index) + 'O.jpg', org)
                #         cv2.imwrite('line1/' + self.filename + str(index) + 'Z.jpg', self.__segment[index])
                # else:
                #     cv2.imwrite('line2/' + self.filename + str(index) + '.jpg', self.__segment[index])
                # combine = False

        self.__segment.reverse()
        print("切割行數：" + str(len(self.__segment)) + "/" + str(len(self.words)) + " 行")


        return len(self.__segment), self.__segment

    def Vertical(self):
        '''
        A function to do vertical projection
        :return:
        '''
        row, col = self.BinImg.shape
        data = verical_pro(self.BinImg)
        th = min(data[10:-10])
        inline = False
        start = 0
        border_limit = 0.5 * row
        lines = []
        for x in range(20, col-10):
            if (not inline) and th+30 < data[x] < border_limit:
                inline = True
                start = x
            elif inline and data[x] < th + 30:
                inline = False
                lines.append([start,x])  # XXXXX
            else:
                pass

        self.__segment = []
        count_which_line = 0
        print(len(lines))
        for line in lines:
            # cv2.imshow('test1', self.BinImg[:, line[0]:line[1]])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if line[1]-line[0] > 20:  # ???調整切割單行參數，最左或最右判斷即可
                print("1")
                if count_which_line == 0 or count_which_line == len(lines)-1:  # 若為最左或最右，則寬度條件變嚴苛
                    if line[1]-line[0] > 30 and np.count_nonzero(self.BinImg[:, line[0]:line[1]]) > 500:
                        self.__segment.append(self.BinImg[:, line[0]:line[1]])
                else:
                    self.__segment.append(self.BinImg[:, line[0]:line[1]])
                # cv2.imshow('test1', self.__segment[-1])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            count_which_line += 1
        self.__segment.reverse()
        print("此頁切出" + str(len(self.__segment)) + "行")
        return len(self.__segment), self.__segment

    def new_CutWord(self):
        global count_name
        # 整頁切出的單字，二維陣列，尚未經過雙行判斷
        lines = []
        # 整頁切出的單字，二維陣列，經過雙行判斷
        words = []
        count_line = 1
        count_word = 0
        for s in self.__segment:
            line_name = self.filename.split(".")[0] + "_" + str(count_line)
            ret, s = cv2.threshold(s, 128, 255, cv2.THRESH_BINARY)
            # cv2.imshow("line", s)
            # cv2.waitKey(0)
            row, col = s.shape
            data = []
            tmp_line = []
            # 找出切字的候選線
            tmp_line.append(0)  # 開頭位置強迫畫候選線
            for i in range(row):
                tmp = np.count_nonzero(s[i, :])
                data.append(tmp)
                # 白點數小於等於2，則畫線
                if tmp <= 2:  # !!!!!!!!!!!!!!!!
                    tmp_line.append(i)
            tmp_line.append(row - 1)  # 結尾位置強迫畫候選線

            line = []
            th = 0.1 * col
            y = 0
            # 從候選線中篩選正確之線段
            while y < len(tmp_line) - 1:
                a = tmp_line[y]
                b = tmp_line[y + 1]
                x = sum(data[a:b])
                if x > th * (b - a):  # 區域內白點數 ?????
                    line.append([a, b, b - a])
                y += 1

            # 避免有兩條線重疊在一起的情形
            for i in range(len(line)-1):
                if line[i][1] == line[i+1][0]:
                    line[i+1][0] += 1

            print("第" + str(count_line) + "行合併前共有 " + str(len(line)) + " 字")

            # 在原圖上標記上下候選線的位置，並存入Global_R跟Global_G中
            Global_R = []
            Global_G = []
            med_candidate = []  # 透過中位數找整行字體大約的高度
            # print(line)
            for l in line:
                med_candidate.append(l[2])
                s[l[0], :] = 122
                Global_R.append(l[0])
                s[l[1], :] = 132
                Global_G.append(l[1])
            # 寫錯位置的過高切割
            # ranges = int(col*0.1)
            # for l in line:
            #     if l[1]-l[0] > col*1.1:
            #         s[l[0], :] = 122
            #         Global_R.append(l[0])
            #         s[l[1], :] = 132
            #         Global_G.append(l[1])
            #         num_words = int((l[1] - l[0]) / col + 1)
            #         for i in range(num_words):
            #             med_candidate.append(int((l[1]-l[0])/num_words))
            #         sep_line = []
            #         if num_words > 1:
            #             for i in range(1, num_words):
            #                 minium = col*255
            #                 minium_line = 0
            #                 for j in range(ranges*-1, ranges+1, 1):
            #                     if minium > cv2.countNonZero(s[l[0] + int((l[1]-l[0])/num_words*i)+j, :]):
            #                         minium = cv2.countNonZero(s[l[0] + int((l[1]-l[0])/num_words*i)+j, :])
            #                         minium_line = l[0] + int((l[1]-l[0])/num_words*i)+j
            #                 sep_line.append(minium_line)
            #         ss = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
            #         for sep in sep_line:
            #             #ss[sep, :] = (0, 0, 255)
            #             s[sep, :] = 132
            #             Global_G.append(sep)
            #             s[sep+1, :] = 122
            #             Global_R.append(sep+1)
            #         cv2.imshow("seg", ss[l[0]:l[1], :])
            #         cv2.waitKey(0)
            #         # cv2.destroyAllWindows()
            #         for i in range(len(sep_line)):
            #             img = cv2.cvtColor(ss[sep_line[i] - ranges * 2:sep_line[i] + (ranges + 1) * 2, :], cv2.COLOR_BGR2GRAY)
            #             ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            #             connectivity=4
            #             output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
            #             labels = output[1]
            #             centroids = output[3]
            #             print(centroids)
            #             print(labels)
            #             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #             img[int(img.shape[0]/2), :] = (128,128,128)
            #             pic_cen = int(img.shape[0]/2)
            #             sep_up = np.zeros(img.shape, np.uint8)
            #             sep_down = np.zeros(img.shape, np.uint8)
            #             for i in range(len(centroids)):
            #                 if i != 0:
            #                     img[int(centroids[i][1]), int(centroids[i][0])] = (0,0,255)
            #                     if centroids[i][1] < pic_cen:
            #                         sep_up[labels==i] = (255,255,255)
            #                     else:
            #                         sep_down[labels==i] = (255,255,255)
            #                     # for j in range(centroids[i][1]):
            #                     #     for k in range(centroids[i][0]):
            #
            #             cv2.namedWindow("cen", cv2.WINDOW_NORMAL)
            #             cv2.resizeWindow("cen", img.shape[1] * 3, img.shape[0] * 3)
            #             cv2.imshow("cen", img)
            #             cv2.waitKey(0)
            #
            #             cv2.namedWindow("up", cv2.WINDOW_NORMAL)
            #             cv2.resizeWindow("up", sep_up.shape[1] * 3, sep_up.shape[0] * 3)
            #             cv2.imshow("up", sep_up)
            #             cv2.waitKey(0)
            #
            #             cv2.namedWindow("down", cv2.WINDOW_NORMAL)
            #             cv2.resizeWindow("down", sep_down.shape[1] * 3, sep_down.shape[0] * 3)
            #             cv2.imshow("down", sep_down)
            #             cv2.waitKey(0)
            #
            #         cv2.destroyAllWindows()
            #         # print(l[1]-l[0], sep_line)
            #         # cv2.imwrite("ya/" + self.filename.split(".")[0] + "_" + str(count_word) + ".jpg", s[l[0]:l[1], :])
            #         # count_word += 1
            #         # cv2.imshow("tmp", s[l[0]:l[1], :])
            #         # cv2.waitKey(0)
            #         # cv2.destroyAllWindows()
            #     else:
            #         med_candidate.append(l[2])
            #         s[l[0], :] = 122
            #         Global_R.append(l[0])
            #         s[l[1], :] = 132
            #         Global_G.append(l[1])
            # Global_R.sort()
            # Global_G.sort()
            # print("Global_R" + str(Global_R))
            # print("Global_G" + str(Global_G))
            # 預估字體之高度
            med = statistics.median(med_candidate)
            # print(med_candidate, med)
            # ss = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
            # ss[s == 122] = (0, 0, 255)
            # ss[s == 132] = (0, 255, 0)
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("test", 50, 900)
            # cv2.imshow('test',ss)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 一行所切出的字，尚未經過雙行判斷
            last_bottom = 0
            seg = []
            for x in range(row):
                if s[x, 0] == 132:
                    last_bottom = x
                if s[x, 0] == 122:
                    # window size中的red、green及red、green的集合
                    r = []
                    g = []
                    all = []
                    # 設定window size
                    next_bottom = 0
                    for k in range(x+1, row):
                        if s[k, 0] == 132:
                            next_bottom = k
                            break
                    # 區域太高要做切割
                    if next_bottom - x > col * 1.1:
                        ranges = int(col*0.1)
                        num_words = int((next_bottom - x) / col + 1)
                        sep_line = []
                        if num_words > 1:
                            for i in range(1, num_words):
                                minium = col * 255
                                minium_line = 0
                                for j in range(ranges * -1, ranges + 1, 1):
                                    if minium > cv2.countNonZero(s[x + int((next_bottom - x) / num_words * i) + j, :]):
                                        minium = cv2.countNonZero(s[x + int((next_bottom - x) / num_words * i) + j, :])
                                        minium_line = x + int((next_bottom - x) / num_words * i) + j
                                sep_line.append(minium_line)
                        ss = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
                        # cv2.imshow("seg", ss[x:next_bottom, :])
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        sep_ups = []
                        sep_downs = []
                        merges = []
                        for i in range(len(sep_line)):
                            img = cv2.cvtColor(ss[sep_line[i] - ranges * 2:sep_line[i] + (ranges + 1) * 2, :],
                                               cv2.COLOR_BGR2GRAY)
                            ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
                            connectivity = 4
                            output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
                            labels = output[1]
                            centroids = output[3]
                            print(centroids)
                            print(labels)
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            img[int(img.shape[0] / 2), :] = (128, 128, 128)
                            pic_cen = int(img.shape[0] / 2)
                            sep_up = np.zeros(img.shape, np.uint8)
                            sep_down = np.zeros(img.shape, np.uint8)
                            for i in range(len(centroids)):
                                if i != 0:
                                    img[int(centroids[i][1]), int(centroids[i][0])] = (0, 0, 255)
                                    if centroids[i][1] < pic_cen:
                                        sep_up[labels == i] = (255, 255, 255)
                                    else:
                                        sep_down[labels == i] = (255, 255, 255)
                                        # for j in range(centroids[i][1]):
                                        #     for k in range(centroids[i][0]):
                            sep_up = cv2.cvtColor(sep_up, cv2.COLOR_BGR2GRAY)
                            ___, sep_up = cv2.threshold(sep_up, 0, 255, cv2.THRESH_BINARY)
                            sep_down = cv2.cvtColor(sep_down, cv2.COLOR_BGR2GRAY)
                            ___, sep_down = cv2.threshold(sep_down, 0, 255, cv2.THRESH_BINARY)
                            sep_ups.append(sep_up)
                            sep_downs.append(sep_down)
                            # cv2.namedWindow("cen", cv2.WINDOW_NORMAL)
                            # cv2.resizeWindow("cen", img.shape[1] * 3, img.shape[0] * 3)
                            # cv2.imshow("cen", img)
                            # cv2.waitKey(0)
                            #
                            # cv2.namedWindow("up", cv2.WINDOW_NORMAL)
                            # cv2.resizeWindow("up", sep_up.shape[1] * 3, sep_up.shape[0] * 3)
                            # cv2.imshow("up", sep_up)
                            # cv2.waitKey(0)
                            #
                            # cv2.namedWindow("down", cv2.WINDOW_NORMAL)
                            # cv2.resizeWindow("down", sep_down.shape[1] * 3, sep_down.shape[0] * 3)
                            # cv2.imshow("down", sep_down)
                            # cv2.waitKey(0)
                        for i in range(len(sep_line)):
                            if len(sep_line) == 1:
                                tmp = np.vstack((s[x: sep_line[i] - ranges * 2, :], sep_ups[0]))
                                merges.append(tmp)
                                tmp = np.vstack((sep_downs[0], s[sep_line[i] + (ranges + 1) * 2:next_bottom, :]))
                                merges.append(tmp)
                                break
                            if i == 0:
                                tmp = np.vstack((s[x:sep_line[i] - ranges * 2, :], sep_ups[0]))
                                merges.append(tmp)
                            elif i == len(sep_line)-1:
                                tmp = np.vstack((sep_downs[i-1], s[sep_line[i-1]+(ranges + 1) * 2: sep_line[i]-ranges * 2, :], sep_ups[i]))
                                merges.append(tmp)
                                tmp = np.vstack((sep_downs[i], s[sep_line[i]+(ranges + 1) * 2:next_bottom, :]))
                                merges.append(tmp)
                            else:
                                tmp = np.vstack((sep_downs[i-1], s[sep_line[i-1]+(ranges + 1) * 2: sep_line[i]-ranges * 2, :], sep_ups[i]))
                                merges.append(tmp)
                        for m in merges:
                            seg.append(m)
                            # cv2.namedWindow("m", cv2.WINDOW_NORMAL)
                            # cv2.resizeWindow("m", m.shape[1] * 3, m.shape[0] * 3)
                            # cv2.imshow("m", m)
                            # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                    # 區域太小要做合併
                    else:
                        if col*0.5 < med < col*1.5:
                            if next_bottom-x > med:
                                WinSize = int(x + med * 1.1)
                            else:
                                if x - int((med - (next_bottom - x)) / 2) <= last_bottom:
                                    WinSize = int(int((x+last_bottom)/2)+med*1.1)
                                else:
                                    WinSize = int(x + med*1.1 - int((med - (next_bottom - x)) / 2))
                        else:
                            WinSize = int(x+col)  # ???col決定window size的方法調整
                        if WinSize > row:
                            WinSize = row
                        # print(med, WinSize-x)
                        # print(last_bottom, x , next_bottom, med)
                        # print("win: " + str(WinSize))
                        # 紀錄window size中所有的候選線
                        for y in range(x, WinSize):
                            if s[y, 0] == 122:
                                r.append(y)
                                all.append(y)
                            elif s[y, 0] == 132:
                                g.append(y)
                                all.append(y)
                            else:
                                pass
                        count = len(all)

                        if count <= 2:
                            seg.append(s[x: Global_G[Global_R.index(x)], :])
                            # cv2.imwrite("ha/" + self.filename.split(".")[0] + "_" + str(count_word) + ".jpg", seg[-1])
                            # count_word += 1
                        elif 2 < count <= 3:
                            seg.append(s[r[0]:g[0], :])
                            # cv2.imwrite("ha/" + self.filename.split(".")[0] + "_" + str(count_word) + ".jpg", seg[-1])
                            # count_word += 1
                        elif count == 4:
                            index = all.index(g[0])
                            if all[index + 1] - g[0] < col * 0.4:
                                s[g[0], :] = 0
                                Global_G.remove(g[0])
                                s[r[1], :] = 0
                                Global_R.remove(r[1])
                                seg.append(s[r[0]:g[1], :])
                                # cv2.imwrite("ha/" + self.filename.split(".")[0] + "_" + str(count_word) + ".jpg", seg[-1])
                                # count_word += 1
                            else:
                                seg.append(s[r[0]:g[0], :])
                                # cv2.imwrite("ha/" + self.filename.split(".")[0] + "_" + str(count_word) + ".jpg", seg[-1])
                                # count_word += 1
                                seg.append(s[r[1]:g[1], :])
                                # cv2.imwrite("ha/" + self.filename.split(".")[0] + "_" + str(count_word) + ".jpg", seg[-1])
                                # count_word += 1
                        else:
                            # 判斷最後一個是紅還是綠線
                            if g[-1] > r[-1]:
                                tmp = all[1:-1]
                                flag = True
                            else:
                                tmp = all[1:-2]
                                flag = False
                            # 將中間所有的線都刪除
                            # print("All" + str(all))
                            # print(tmp)
                            for t in tmp:
                                if s[t, 0] == 122:
                                    Global_R.remove(t)
                                elif s[t, 0] == 132:
                                    Global_G.remove(t)
                                else:
                                    pass
                                s[t, :] = 0
                            if flag:
                                seg.append(s[all[0]:all[-1], :])
                            else:
                                seg.append(s[all[0]:all[-2], :])
                            # cv2.imwrite("ha/" + self.filename.split(".")[0] + "_" + str(count_word) + ".jpg", seg[-1])
                            # count_word += 1
            lines.append(seg.copy())
            print("第" + str(count_line) + "行合併後共有 " + str(len(seg)) + " 字")
            count_line += 1
            # cv2.imwrite("yolo/lines/" + line_name + ".jpg", s)
            # for se in seg:
            #     cv2.imshow('test', se)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # ss = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
            # ss[s == 122] = (0, 0, 255)
            # ss[s == 132] = (0, 255, 0)
        count_line = 0
        print("雙行判斷前共有 " + str(len(lines)) + " 行")
        for l in lines:
            count_line += 1
            line_name = self.filename.split(".")[0] + "_" + str(count_line)
            tmp = []
            tmp_left = []
            last_is_two = False
            if head_tail_not_a_word(l[0]):
                del l[0]
            if len(l) > 0 and head_tail_not_a_word(l[-1]):
                del l[-1]
            count_word = 0
            for w in l:
                # count_word += 1
                # cv2.imshow('test', w)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if not two_word(w):
                    if last_is_two and tmp != []:
                        words.append(tmp.copy())
                        tmp = tmp_left
                        tmp_left = []
                    tmp.append(normalize(w))
                    # cv2.imshow("w", w)
                    # cv2.waitKey(0)
                    # cv2.imwrite("yolo/words/" + line_name + "_" + str(count_word) + ".jpg", w)
                    # count_word+=1
                    # cv2.imwrite("1383BG/"+"0_" + self.filename[2:8] + "_" + self.filename[8:-4] + "_" + str(
                    #         count_line) + "_S_" + str(count_word)+".jpg", normalize(w))
                    # count_word+=1
                    last_is_two = False
                else:
                    last_is_two = True
                    mid = two_word_mid(w)
                    if mid != 0:
                        tmp_left.append(normalize(w[:, 0:mid]))
                        tmp.append(normalize(w[:, mid:-1]))
                        # cv2.imshow("w", w[:, 0:mid])
                        # cv2.waitKey(0)
                        # cv2.imshow("w", w[:, mid:-1])
                        # cv2.waitKey(0)
                        # cv2.imwrite("yolo/words/" + line_name + "_" + str(count_word) + ".jpg", w[:, 0:mid])
                        # count_word += 1
                        # cv2.imwrite("yolo/words/" + line_name + "_" + str(count_word) + ".jpg", w[:, mid:-1])
                        # count_word += 1
                        # cv2.imwrite("1383BG/" + "0_" + self.filename[2:8] + "_" + self.filename[8:-4] + "_" + str(
                        #     count_line) + "_D_L" + str(count_word) + ".jpg", normalize(w[:, 0:mid]))
                        # count_word+=1
                        # cv2.imwrite("1383BG/"+"0_" + self.filename[2:8] + "_" + self.filename[8:-4] + "_" + str(
                        #     count_line) + "_D_R" + str(count_word)+".jpg", normalize(w[:, mid:-1]))
                        # count_word+=1
                    else:
                        tmp.append(normalize(w))
                        # cv2.imshow("w", w)
                        # cv2.waitKey(0)
                        # cv2.imwrite("yolo/words/" + line_name + "_" + str(count_word) + ".jpg", w)
                        # count_word+=1
                        # cv2.imwrite("1383BG/" + "0_" + self.filename[2:8] + "_" + self.filename[8:-4] + "_" + str(
                        #     count_line) + "_S_" + str(count_word) + ".jpg", normalize(w))
                        # count_word+=1
                        last_is_two = False
            if tmp != []:
                words.append(tmp.copy())
            if tmp_left != []:
                words.append(tmp_left.copy())
        print("雙行判斷後共有 " + str(len(words)) + " 行")

        # for li in words:
        #     for w in li:
        #         cv2.imshow('YZYZY', w )
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()
        for j in range(len(words)):
            a = words[j][0]
            words[j][0][-3:-1, :] = 128
            for i in range(1, len(words[j])):
                words[j][i][-3:-1, :] = 128
                a = np.vstack((a, words[j][i]))
            cv2.namedWindow("0", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("0", 10, len(words[j])*45)
            cv2.imshow("0", a)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return words


    def CutWord(self):
        global count_name, one, two, three, four, pre_one, pre_two, pre_three, pre_four
        word = []
        word_show = []
        # hist = horizontal_pro(self.BinImg)

        for s in self.__segment:
            previous_line = len(word)
            ret, s = cv2.threshold(s, 128, 255, cv2.THRESH_BINARY)
            text = []
            data = []
            tmp_line = []
            row, col = s.shape
            tmp_line.append(0)
            for i in range(row):
                tmp = np.count_nonzero(s[i, :])
                data.append(tmp)
                if tmp == 0:
                    tmp_line.append(i)
            tmp_line.append(row-1)  # 避免有字貼到底部卻未被切割的情形

            line = []
            th = 0.1 * col
            y = 0
            while y < len(tmp_line) - 1:
                a = tmp_line[y]
                b = tmp_line[y + 1]
                x = sum(data[a:b])
                if x > th * (b - a):  # 區域內白點數  and (col not in data[a:b]) → 判斷邊界
                    line.append([a, b, b - a, x])
                y += 1

            for i in range(len(line)-1):
                if line[i][1] == line[i+1][0]:
                    line[i+1][0] += 1
            Global_R = []
            Global_G = []
            med_candidate = []
            for l in line:
                med_candidate.append(l[2])
                s[l[0], :] = 122
                Global_R.append(l[0])
                s[l[1], :] = 132
                Global_G.append(l[1])
                # cv2.imwrite("mix/" + str(count_name)+".jpg", s[l[0]:l[1], :])
                # count_name += 1
                # cv2.imshow('test', s[l[0]:l[1],:])
                # cv2.waitKey(0)
                # cv2.destroyWindow('test')
            med = statistics.median(med_candidate)
            print("此行切出" + str(len(Global_G)) + "區塊")
            ss = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
            ss[s == 122] = (0, 0, 255)
            ss[s == 132] = (0, 255, 0)
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("test", 50, 900)
            # cv2.imshow('test',ss)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print("此行編號： " + str(count_name))
            # cv2.imwrite(str(count_name)+".jpg", ss)  # 尚未進行雙行切割及合併的單字切割影像
            count_name += 1
            line1 = []
            line2 = []
            seg = []
            seg_right = []
            seg_left = []
            # first = True
            two_word_line = False
            head = 0  # 記錄從單行或雙行開始
            last_bottom = 0
            for x in range(row):
                if s[x, 0] == 132:
                    last_bottom = x
                elif s[x, 0] == 122:
                    r = []
                    g = []
                    all = []
                    ##########
                    # 設定window size
                    # next_bottom = 0
                    # for k in range(x + 1, row):
                    #     if s[k, 0] == 132:
                    #         next_bottom = k
                    #         break
                    if col * 0.5 < med < col * 1.5:
                        if x > med:
                            WinSize = int(x + med * 1.1)
                        else:
                            if x - int((med - (x - last_bottom)) / 2) <= last_bottom:
                                WinSize = int(int((x + last_bottom) / 2) + med * 1.1)
                            else:
                                WinSize = int(x + int((med - (x - last_bottom)) / 2) * 1.1 + (x - last_bottom))
                    else:
                        WinSize = int(x + col * 1.2)  # ???window size調整
                    if WinSize > row:
                        WinSize = row
                    ##############
                    for y in range(x, WinSize):
                        if s[y, 0] == 122:
                            r.append(y)
                            all.append(y)
                        elif s[y, 0] == 132:
                            g.append(y)
                            all.append(y)
                        else:
                            pass

                    count = len(all)
                    # print(count)
                    # print("count: ", count)
                    if count <= 2:
                        one += 1
                        if not two_word(s[x: Global_G[Global_R.index(x)], :]):
                            seg.append(s[x: Global_G[Global_R.index(x)], :])
                            if head == 0:
                                two_word_line = False
                                head = 1
                            if two_word_line and seg_left != [] and seg_right != []:
                                line2.append((seg_left.copy(), seg_right.copy()))
                                seg_left.clear()
                                seg_right.clear()
                                two_word_line = False
                        else:
                            mid = two_word_mid(s[x: Global_G[Global_R.index(x)], :])
                            seg_left.append(s[x: Global_G[Global_R.index(x)], 0:mid])
                            seg_right.append(s[x: Global_G[Global_R.index(x)], mid:-1])
                            # print("SEG")
                            if head == 0:
                                two_word_line = True
                                head = 2
                            if not two_word_line and seg != []:
                                line1.append(seg.copy())
                                seg.clear()
                                two_word_line = True
                    elif 2 < count <= 3:
                        one += 1
                        if two_word(s[r[0]:g[0], :]):
                            mid = two_word_mid(s[r[0]:g[0], :])
                            seg_left.append(s[r[0]:g[0], 0:mid])
                            seg_right.append(s[r[0]:g[0], mid:-1])
                            if head == 0:
                                two_word_line = True
                                head = 2
                            # print("SEG")
                            if not two_word_line and seg != []:
                                line1.append(seg.copy())
                                seg.clear()
                                two_word_line = True
                        else:
                            seg.append(s[r[0]:g[0], :])
                            if head == 0:
                                two_word_line = False
                                head = 1
                            if two_word_line and seg_left != [] and seg_right != []:
                                line2.append((seg_left.copy(), seg_right.copy()))
                                seg_left.clear()
                                seg_right.clear()
                                two_word_line = False
                    elif count == 4:
                        # cv2.imshow('test',s[r[0]:g[1], :])
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        index = all.index(g[0])
                        if all[index + 1] - g[0] < col * 0.4 and (not two_word(s[r[0]:g[0], :]) or not two_word(s[r[1]:g[1], :])):
                            three += 1
                            s[g[0], :] = 0
                            Global_G.remove(g[0])
                            s[r[1], :] = 0
                            Global_R.remove(r[1])
                            seg.append(s[r[0]:g[1], :])
                            # print("merge two!")
                            if head == 0:
                                two_word_line = False
                                head = 1
                            if two_word_line and seg_left != [] and seg_right != []:
                                line2.append((seg_left.copy(), seg_right.copy()))
                                seg_left.clear()
                                seg_right.clear()
                                two_word_line = False
                        elif two_word(s[r[0]:g[0], :]) and two_word(s[r[1]:g[1], :]):
                            two += 1
                            mid = two_word(s[r[0]:g[0], :])
                            seg_left.append(s[r[0]:g[0], 0:mid])
                            seg_right.append(s[r[0]:g[0], mid:-1])
                            mid = two_word_mid(s[r[1]:g[1], :])
                            seg_left.append(s[r[1]:g[1], 0:mid])
                            seg_right.append(s[r[1]:g[1], mid:-1])
                            s[r[0], :] = 0
                            s[g[0], :] = 0
                            s[r[1], :] = 0
                            s[g[1], :] = 0
                            # print("SEG TWO")
                            if head == 0:
                                two_word_line = True
                                head = 2
                            if not two_word_line and seg != []:
                                line1.append(seg.copy())
                                seg.clear()
                                two_word_line = True
                    elif count <= 7:  # 尚未處理此情形，僅全部合併
                        four += 1
                        if g[-1] > r[-1]:
                            tmp = all[1:-1]
                            flag = True
                        else:
                            tmp = all[1:-2]
                            flag = False
                        # print(tmp)
                        for t in tmp:
                            if s[t, 0] == 122:
                                Global_R.remove(t)
                            elif s[t, 0] == 132:
                                Global_G.remove(t)
                            else:
                                pass
                            s[t, :] = 0
                        if flag:
                            # cv2.imshow("tmp", s[all[0]:all[-1], :])
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            seg.append(s[all[0]:all[-1], :])
                        else:
                            # cv2.imshow("tmp", s[all[0]:all[-2], :])
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            seg.append(s[all[0]:all[-2], :])
                        # cv2.imshow("tmp", s[tmp[0]:tmp[1], :])
                        # cv2.waitKey(0)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        if head == 0:
                            two_word_line = False
                            head = 1
                        if two_word_line and seg_left != [] and seg_right != []:
                            line2.append((seg_left.copy(), seg_right.copy()))
                            seg_left.clear()
                            seg_right.clear()
                            two_word_line = False
            if seg:
                line1.append(seg.copy())
                seg.clear()
            elif seg_left != [] and seg_right != []:
                line2.append((seg_left.copy(), seg_right.copy()))
                seg_left.clear()
                seg_right.clear()

            while line1 != [] or line2 != []:
                if head == 1:
                    for i in range(len(line1[0])):
                        if 0 not in line1[0][i].shape:  # 去除長寬其一為0的圖片
                            # cv2.imshow("123", line1[0][i].copy())
                            # cv2.waitKey(0)
                            text.append(line1[0][i].copy())
                    del line1[0]
                    head = 2
                elif head == 2:
                    for i in range(len(line2[0][1])):
                        if 0 not in line2[0][1][i].shape:
                            # cv2.imshow("456", line2[0][1][i].copy())
                            # cv2.waitKey(0)
                            text.append(line2[0][1][i].copy())
                    word.append(text.copy())
                    text.clear()
                    for i in range(len(line2[0][0])):
                        if 0 not in line2[0][0][i].shape:
                            # cv2.imshow("789", line2[0][0][i].copy())
                            # cv2.waitKey(0)
                            text.append(line2[0][0][i].copy())
                    del line2[0]
                    head = 1
                # cv2.destroyAllWindows()
            if text:
                word.append(text.copy())
                text.clear()
            print("此行被分割成" + str(len(word)-previous_line) + "行" + " 分別為")
            for tmp in word[previous_line:len(word)]:
                print(str(len(tmp))+"字")
            print("一區塊：" + str(one-pre_one))
            print("兩區塊(兩雙)： " + str(two-pre_two))
            print("兩區塊(一單一雙或兩單)： " + str(three-pre_three))
            print("三區塊以上： " + str(four-pre_four) + "\n")
            pre_one = one
            pre_two = two
            pre_three = three
            pre_four = four

        print("去頭尾： ")
        for i in range(len(word)):
            if head_tail_not_a_word(word[i][0]):
                del word[i][0]
            if len(word[i]) >= 1 and head_tail_not_a_word(word[i][-1]):
                del word[i][-1]
            for j in range(len(word[i])):
                word[i][j] = normalize(word[i][j])
                # cv2.imshow("final", word[i][j])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            print(len(word[i]))
            # print(len(word))

        global count_two_word
        # print("雙字數量：" + str(count_two_word))

        # for j in range(len(word)):
        #     print(len(word))
        #     a = word[j][0]
        #     for i in range(1, len(word[j])):
        #         a = np.vstack((a, word[j][i]))
        #     cv2.namedWindow("0", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("0", 30, len(word[j])*30)
        #     cv2.imshow("0", a)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
            # print(" ")

        return word

def head_tail_not_a_word(img):
    h, w = img.shape
    if h*w <= 0:
        return True
    # cv2.imshow("bad", img)
    # print(img.shape)
    ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    hor_hist = horizontal_pro(img)
    if h*3 > w:
        # print("False: h*3>w", h, w)
        # cv2.imshow("bad", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return False
    for line in hor_hist:
        if line >= 0.9*w:
            # print("True: line > 0.9*w", line, w)
            # cv2.imshow("bad", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return True
    # connected component
    stats, labels = cv2.connectedComponents(img)
    if stats > 3:
        # print("component over 3")
        # cv2.imshow("bad", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return True

    # cv2.imshow("bad", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return False

count_two_word = 0
def two_word_mid(img):
    global count_two_word
    h, w = img.shape
    hist = []
    for i in range(w):
        hist.append(np.count_nonzero(img[:, i]))

    hor_start = 0
    hor_end = w - 1
    for i in range(len(hist)):
        if hist[i] > 1:
            hor_start = i
            break
    hist.reverse()
    for i in range(len(hist)):
        if hist[i] > 1:
            hor_end = w - i
            break
    hist.reverse()

    candidate_mid = []
    start = -1
    end = -2
    for i in range(len(hist)):
        if hist[i] < h * 0.05 and start == -1:
            start = i
        elif hist[i] < h * 0.05:
            pass
        elif hist[i] >= h * 0.05 and start != -1 and end == -2:
            end = i
            if end - start >= 3:
                candidate_mid.append(int((start + end) / 2))
            start = -1
            end = -2
    hor_mid = int((hor_end + hor_start) / 2)
    # print(int(w/2), hor_mid)
    for i in candidate_mid:
        if hor_mid * 0.8 < i < hor_mid * 1.2:
            count_two_word += 1
            return i
    return 0

count_name1 = 0
count_name2 = 0
count_name = 0
def two_word(img):
    # return False
    # cv2.imshow('test', img)
    # img_tmp = cv2.resize(img, (80, 45))
    # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2GRAY)
    img_tmp = normalize_cnn(img)
    # __, img_tmp = cv2.threshold(img_tmp, 128, 255, cv2.THRESH_BINARY_INV)  # 改成白底黑字
    img_tmp = np.expand_dims(img_tmp, axis=-1)
    img_tmp = np.expand_dims(img_tmp, axis=0)
    img_tmp = img_tmp / 255
    y = model.predict(img_tmp).tolist()
    # print(y[0].index(max(y[0])))
    if y[0].index(max(y[0])) == 0 or y[0].index(max(y[0])) == 2:
        # print("one")
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('test')
        return False
    else:
        # print("two")
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('test')
        return True

    global count_name1, count_name2, count_name
    h, w = img.shape
    # if h*1.2 < w:
    #     cv2.imwrite("testing/tmp/tmp_double_side/EJJ_"+str(count_name)+".jpg", img)
    #     count_name+=1
    hist1 = []
    hist2 = []
    # ver_show = np.zeros((h,w), np.uint8)
    for i in range(w):
        hist1.append(np.count_nonzero(img[:, i]))
        # ver_show[h-hist1[-1]:h, i] = 255
    # print(hist)
    for i in range(h):
        hist2.append(np.count_nonzero(img[i, :]))
    # print(h,hist2)

    hor_start = 0
    hor_end = w-1
    for i in range(len(hist1)):
        if hist1[i] > 1:
            hor_start = i
            break
    hist1.reverse()
    for i in range(len(hist1)):
        if hist1[i] > 1:
            hor_end = w-i
            break
    hist1.reverse()

    ver_start = 0
    ver_end = h-1
    for i in range(len(hist2)):
        if hist2[i] > 1:
            ver_start = i
            break
    hist2.reverse()
    for i in range(len(hist2)):
        if hist2[i] > 1:
            ver_end = h-i
            break
    hist2.reverse()

    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('test')

    # print(ver_end-ver_start, hor_end - hor_start)
    if not (ver_end-ver_start)*1.2 < hor_end - hor_start < (ver_end-ver_start)*5:  # ????長寬還需再衡量
        # print("W problem")
        return False

    # cv2.imshow("two", img)
    # cv2.waitKey(0)
    candidate_mid = []
    start = -1
    end = -2
    for i in range(len(hist1)):
        if hist1[i] < h*0.05 and start == -1:
            start = i
        elif hist1[i] < h*0.05:
            pass
        elif hist1[i] >= h*0.05 and start != -1 and end == -2:
            end = i
            if end - start >= 3:
                candidate_mid.append(int((start+end)/2))
            start = -1
            end = -2
    hor_mid = int((hor_end+hor_start)/2)
    # print(int(w/2), hor_mid)
    middle_blank = False
    for i in candidate_mid:
        if hor_mid*0.8 < i < hor_mid*1.2:
            middle_blank = True
            # cv2.imwrite("half/" + str(count_name)+".jpg", img)  # 雙行
            # count_name += 1
            # count_name1 += 1
            # print("two")
            # cv2.destroyAllWindows()
    if not middle_blank:
        # cv2.imwrite("testing/tmp/tmp_single/EJJ_"+str(count_name)+".jpg", img)
        # count_name += 1
        return False
    # 比較左右大小比例
    else:
        # cv2.imwrite("testing/tmp/tmp_double/EJJ_"+str(count_name)+".jpg", img)
        # count_name += 1
        # mid = two_word_mid(img)
        # print("mid: " + str(h), str(w), str(mid))
        # left_img = img[:, 0:mid]
        # right_img = img[:, mid:-1]
        # # kernel = np.ones((3, 3), np.uint8)
        # # right_img = cv2.morphologyEx(right_img, cv2.MORPH_OPEN, kernel)
        # # left_img = cv2.morphologyEx(left_img, cv2.MORPH_OPEN, kernel)
        #
        # left_h, left_w = left_img.shape
        # right_h, right_w = right_img.shape
        # left_hist1 = []
        # left_hist2 = []
        # right_hist1 = []
        # right_hist2 = []
        # # 左圖垂直投影
        # for i in range(left_w):
        #     left_hist1.append(np.count_nonzero(left_img[:, i]))
        # # 左圖水平投影
        # for i in range(left_h):
        #     left_hist2.append(np.count_nonzero(left_img[i, :]))
        # # 右圖垂直投影
        # for i in range(right_w):
        #     right_hist1.append(np.count_nonzero(right_img[:, i]))
        # # 右圖水平投影
        # for i in range(right_h):
        #     right_hist2.append(np.count_nonzero(right_img[i, :]))
        #
        # left_hor_start = 0
        # left_hor_end = left_w-1
        # left_ver_start = 0
        # left_ver_end = left_h-1
        # right_hor_start = 0
        # right_hor_end = right_w-1
        # right_ver_start = 0
        # right_ver_end = right_h-1
        # # 左圖
        # for i in range(len(left_hist1)):
        #     if left_hist1[i] > 1:
        #         left_hor_start = i
        #         break
        # left_hist1.reverse()
        # for i in range(len(left_hist1)):
        #     if left_hist1[i] > 1:
        #         left_hor_end = left_w - i
        #         break
        # left_hist1.reverse()
        #
        # for i in range(len(left_hist2)):
        #     if left_hist2[i] > 1:
        #         left_ver_start = i
        #         break
        # left_hist2.reverse()
        # for i in range(len(left_hist2)):
        #     if left_hist2[i] > 1:
        #         left_ver_end = left_h - i
        #         break
        # left_hist2.reverse()
        #
        # # 右圖
        # for i in range(len(right_hist1)):
        #     if right_hist1[i] > 1:
        #         right_hor_start = i
        #         break
        # right_hist1.reverse()
        # for i in range(len(right_hist1)):
        #     if right_hist1[i] > 1:
        #         right_hor_end = right_w - i
        #         break
        # right_hist1.reverse()
        #
        # for i in range(len(right_hist2)):
        #     if right_hist2[i] > 1:
        #         right_ver_start = i
        #         break
        # right_hist2.reverse()
        # for i in range(len(right_hist2)):
        #     if right_hist2[i] > 1:
        #         right_ver_end = right_h - i
        #         break
        # right_hist2.reverse()
        #
        # # 長寬比、長度比、寬度比、周長
        # print(left_hor_start, left_hor_end, left_hor_end-left_hor_start)
        # print(left_ver_start, left_ver_end, left_ver_end-left_ver_start)
        # print(right_hor_start, right_hor_end, right_hor_end-right_hor_start)
        # print(right_ver_start, right_ver_end, right_ver_end-right_ver_start)
        # print("左圖長寬比：" + str((left_hor_end-left_hor_start)/(left_ver_end-left_ver_start)))
        # print("右圖長寬比：" + str((right_hor_end-right_hor_start)/(right_ver_end-right_ver_start)))
        # print("長度比：" + str((left_hor_end-left_hor_start)/(right_hor_end-right_hor_start)))
        # print("寬度比：" + str((left_ver_end-left_ver_start)/(right_ver_end-right_ver_start)))
        # cv2.imshow("left", left_img)
        # cv2.imshow("right", right_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return True
    # cv2.destroyAllWindows()
    # cv2.imwrite("half/" + str(count_name)+".jpg", img)
    # count_name += 1
    # count_name2 += 1
    # print("one")

def LCS(a, b):
    '''
    :param a: answer
    :param b: test_data
    :return: right , wrong
    '''
    len1, len2 = len(a) + 1, len(b) + 1
    arr = [[0 for x in range(len1)] for y in range(len2)]
    # print (arr)
    for y in range(1, len2):
        for x in range(1, len1):
            if a[x - 1] == b[y - 1]:
                arr[y][x] = arr[y - 1][x - 1] + 1
            else:
                arr[y][x] = max(arr[y - 1][x], arr[y][x - 1])

    LCS_size = arr[len(b)][len(a)]
    return LCS_size, len(a) - LCS_size



def normalize(img_gray):
    __, img = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    # print(img.shape)
    h, w = img.shape
    for i in range(h):
        if np.count_nonzero(img[i, :]) != 0:
            img = img[i:h, :]
            break
    h, w = img.shape
    for i in range(h-1, -1, -1):
        if np.count_nonzero(img[i, :]) != 0:
            img = img[0:i+1, :]
            break
    h, w = img.shape
    for i in range(w):
        if np.count_nonzero(img[:, i]) != 0:
            img = img[:, i:w]
            break
    h, w = img.shape

    for i in range(w-1, -1, -1):
        if np.count_nonzero(img[:, i]) != 0:
            img = img[:, 0:i+1]
            break
    h, w = img.shape

    # cv2.imshow("show", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 將短邊補長
    if h < w:
        h1 = int((w-h)/2)
        h2 = (w-h)-h1
        padding1 = np.zeros((h1, w), np.uint8)
        padding2 = np.zeros((h2, w), np.uint8)
        img = np.vstack((padding1, img, padding2))
    elif h > w:
        w1 = int((h-w)/2)
        w2 = (h-w)-w1
        padding1 = np.zeros((h, w1), np.uint8)
        padding2 = np.zeros((h, w2), np.uint8)
        img = np.hstack((padding1, img, padding2))
    # 重新resize成50*50，其中外框有5 pixels的border
    nor = cv2.resize(img, (60,60), interpolation=cv2.INTER_CUBIC)
    hor_border = np.zeros((60,2), np.uint8)
    ver_border = np.zeros((2,64), np.uint8)
    nor = np.hstack((hor_border, nor, hor_border))
    nor = np.vstack((ver_border, nor, ver_border))
    __, finish = cv2.threshold(nor, 85, 255, cv2.THRESH_BINARY)
    # cv2.imshow("tmp", finish)
    # print (finish.shape)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("tmp/"+file.split(".")[0]+"_after"+".jpg", finish)
    return finish

def normalize_cnn(img):
    h, w = img.shape
    # 假定圖片大小要變成 85*40
    ratio = w / h  # 比例為 2.125
    if ratio <= 2.125:
        # if h < w:
        x_dif = (2.125 * h) - w  # 找出寬還差多少
        x_pad = int(x_dif / 2)
        padding = np.zeros((h, x_pad), np.uint8)
        # padding = padding + 255
        # print("h: {}, w: {}, x_pad: {}, ratio: {}".format(h, w, x_pad, ratio))
        img = np.hstack((padding, img, padding))

    elif ratio > 2.125:
        y_dif = (w / 2.125) - h  # 找出高還差多少 padding 才能使圖片比例為 2.125
        y_pad = int(y_dif / 2)
        # print("h: {}, w: {}, y_pad: {}, ratio: {}".format(h, w, y_pad, ratio))
        padding = np.zeros((y_pad, w), np.uint8)
        # padding = padding + 255
        img = np.vstack((padding, img, padding))
    img = cv2.resize(img, (85, 40), interpolation=cv2.INTER_CUBIC)
    return img

# 以下為tesseract的測試函式
# if __name__ == '__main__':
#     filename = []
#     context = []
#     page = {}
#     with open('test_data.txt','r',encoding='utf-8') as fp:
#         for lines in fp:
#             string = lines.split('\n')[0]
#             if string.find('DD')!=-1:
#                 if context!=[]:
#                     page[filename] = context
#                     context = []
#                 filename = string
#             else:
#                 context.append(string)
#     fp.close()
#     count = 0
#     length = len(page)
#     right = 0
#     wrong = 0
#     real_file = []
#     for f, arr in page.items():
#         print (f)
#         count +=1
#         src = image(f+'.jpg')
#         src.pre_process()
#         cv2.imwrite('tmp1.jpg',~src.BinImg)
#         val , s = src.Vertical()
#         print (count , "/" , length)
#         if val == len(arr):
#             word = src.CutWord()
#             # cwd = os.getcwd()
#             for i in range(len(word)):
#                  if len(word[i]) == len(arr[i]):
#                      size = len(word[i])
#                      for x in range(size):
#                          img = word[i][x]
#                          filedir = arr[i][x]
#                          cv2.imwrite('tmp1.jpg',img)
#                          detect = pytesseract.image_to_string(Image.open('tmp1.jpg'), lang='chi_tra' ,config='--psm 10 --oem 3')
#                          if detect == filedir:
#                             right+=1
#                          else:
#                             wrong+=1
#                          print("Now : {} , {}".format(right,wrong))
#                          print(right/(right+wrong))
#                  else:
#                     print("No, Wrong word")
#         else:
#             print("No, Wrong line")


# if __name__ == "__main__":
    # filename = []
    # context = []
    # pages = []
    # with open('test_data2.txt','r',encoding='utf-8') as fp:
    #     for lines in fp:
    #         string = lines.split('\n')[0]
    #         if string.find('DD')!=-1:
    #             if context!=[]:
    #                 pages.append([filename,context])
    #                 context = []
    #             filename = string
    #         else:
    #             context.append(string)
    # fp.close()
    #
    # right = 0
    # count_tesseract_word = 0
    # count_page_word = 0
    # wrong = 0
    # size = len(pages)
    # for i , page in enumerate(pages):
    #     print ("%d / %d " % (i+1,size))
    #     ocr = []
    #     count_ocr_word = 0
    #     key = page[0]
    #     if key.find('.jpg')!=-1:
    #         continue
    #     items = page[1]
    #     src = image(key+'.jpg')
    #     src.pre_process()
    #     col_count , col_data = src.Vertical()
    #     if col_count==len(items):
    #         for item in items:
    #             count_page_word += len(item)
    #         print (key)
    #         col_words = src.CutWord()
    #         # for words in col_words:
    #         #     string = ''
    #         #     for word in words:
    #         #         cv2.imwrite('tmp.jpg',word)
    #         #         string += pytesseract.image_to_string(Image.open('tmp.jpg'),lang='chi_tra',config='--oem 3 --psm 10')
    #         #     count_tesseract_word+=len(string)
    #         #     ocr.append(string)
    #         # for index , item in enumerate(items):
    #         #     right_tmp , wrong_tmp = LCS(item,ocr[index])
    #         #     right += right_tmp
    #     else:
    #         wrong +=1
    #     print ("Recall : %d / %d" % (right , count_page_word ))
    #     print ("Pression : %d / %d" % ( right , count_tesseract_word ))
    #     print ("now wrong page count %d " % (wrong))