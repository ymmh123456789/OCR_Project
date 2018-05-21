import cv2
import os

if __name__ == "__main__":
    for i in range(8):
        print(i)
    file = "123456789_1"
    filename = "123456789_1_2"
    print(filename.split("_")[0] + "_" + filename.split("_")[1])
    # os.chdir("one")
    # filenames = os.listdir(os.getcwd())
    # for file in filenames:
    #     img = cv2.imread(file, 0)
    #     img = image.normalize(img)
    #     cv2.imwrite(file, img)
        # cv2.imshow("123", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # count = 138
    # os.chdir("EJJ")
    # filenames = os.listdir(os.getcwd())
    # count = 0
    # for filename in filenames:
    #     count+=1
    #     tmp = filename.split("_")
    #     if tmp[5][0] == "L":
    #         new = tmp[0] + '_' + tmp[1] + "_" + tmp[2] + "_" + tmp[3] + "_" + "L_" + tmp[5][1:]
    #     elif tmp[5][0] == "R":
    #         new = tmp[0] + "_" + tmp[1] + "_" + tmp[2] + "_" + tmp[3] + "_" + "R_" + tmp[5][1:]
    #     else:
    #         new = tmp[0] + "_" + tmp[1] + "_" + tmp[2] + "_" + tmp[3] + "_" + "M_" + tmp[5]
    #     # print(filename, new)
    #     # tmp = tmp1 + "8" + tmp2
    #     if filename != new:
    #         os.rename(filename, new)  # tmp
    #     if count%10==0:
    #         print(str(count) + "/755915")
    #     # count+=1