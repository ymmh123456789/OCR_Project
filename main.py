# import numpy as np
import os
# import argparse
import cv2

# import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument('model',type=str, help='Choose model')
# parser.add_argument('folder',type=str, help='input folder')
# args = parser.parse_args()
#
# if args.model==None or args.folder==None:
#     print('python main.py --folder=<Your folder> --model=<model>')
#     exit(-1)

# read model
# with open('book1_map.pkl','rb') as reader:
#     map = pickle.load(reader)
# reader.close()
from image import image

# input_folder = 'book7/double/'
input_folder = 'book7/single/'

# files = os.listdir(input_folder+"/PDF_to_JPG")
# from keras.models import load_model
# model = load_model('Book1.h5')
# CNN辨識
# for file in files:
#     if file.split('.')[-1]!='jpg':
#         continue
#     src = image(input_folder+"/PDF_to_JPG" + '/'+ file)
#     src.pre_process()
#     src.Vertical()
#     all_word_in_page = src.CutWord()

    # str = ''
    # for col in all_word_in_page:
    #     if col == []:
    #         continue
    #     all_col_word = np.array(col)
    #     all_col_word = all_col_word.reshape(all_col_word.shape[0],64,64,1)
        # pred = model.predict_classes(all_col_word).tolist()
        # for p in pred:
        #     str+=map[p]
        # str+='\n'
    # with open(input_folder+'/'+file+'.txt','w',encoding='utf-8') as writer:
    #     writer.writelines(str)
    # writer.close()

# 比較行列
filename = []
context = []
pages = []
with open(input_folder + 'match.txt', 'r', encoding='utf-8') as fp:
    for line in fp:
        string = line.split('\n')[0]
        if string.find('DD') != -1:
            if context != []:
                pages.append([filename, context])
                context = []
            filename = string.replace("\ufeff", "")
        else:
            string = string.replace("\uf470", "").replace("\uf6a4", "")
            context.append(string)
if context != []:
    pages.append([filename, context])

files = []
for subdir, dirs, tmp_files in os.walk(input_folder+"/PDF_to_JPG"):
    for file in tmp_files:
        files.append(file.split(".")[0])

fp.close()
wrong = 0
right = 0
count_page = 0
count_word = 0
count_line = 0
count_right_line = 0    # 行數且字數正確
count_wrong_line = 0    # 該頁行數錯誤
# del pages[0]
for page in pages:
    print(page[0].split(".")[0])  # , right, "/", count_page
    # print(page[1])
    if page[1] == [] or len(page[0].split(".")) > 1 or page[0] not in files:
        print("Not Exist!")
        continue
    # count_page += 1
    # if count_page < 2642:
    #     continue
    p = image(input_folder + "PDF_to_JPG" + "/" + page[0].split(".")[0]+'.jpg')
    # p = image("DD1379BX3000021-101.jpg")
    p.pre_process()
    p.new_Vertical()
    img = p.new_CutWord()
    for words in page[1]:
        count_word += len(words)
    count_line += len(page[1])
    print("正確行數： " + str(len(page[1])))
    print("切割行數： " + str(len(img)))
    if len(page[1]) == len(img):  # 行數正確
        count = 0
        for i in range(len(page[1])):
            if len(page[1][i]) == len(img[i]):
                count += 1
        count_right_line += count
        if count == len(page[1]):  # 字數正確
            # print(page[0].split(".")[0])
            # right+=1
            with open(input_folder + "page_with_right_count.txt", "a", encoding='utf-8') as W:
                W.writelines(page[0]+"\n")
                for i in range(len(page[1])):
                    W.writelines(page[1][i]+"\n")
            W.close()
    else:
        count_wrong_line+=len(page[1])
    # print("lines: ", len(img))
    # for p in img:
    #     print(len(p))
    print("總行數： "+str(count_line))
    print("行數錯誤之行數： "+str(count_wrong_line))
    print("行數正確但字數錯誤之行數： "+str(count_line-count_wrong_line-count_right_line))
    print("行數正確且字數正確之行數： "+str(count_right_line))
    print("總字數："+str(count_word) + "\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()