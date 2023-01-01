import csv
import shutil

root = r'C:/Users/lenovo/Desktop/mobile_vit/bitmoji_data/trainimages/'
female = r'C:/Users/lenovo/Desktop/mobile_vit/bitmoji_data/train/female/'
male = r'C:/Users/lenovo/Desktop/mobile_vit/bitmoji_data/train/male/'



def csv_read():
    """读取csv文件"""
    with open(r'C:\Users\lenovo\Desktop\mobile_vit\data_set\Bitmojidata\train.csv ', encoding='utf8') as f:
        reader=csv.reader(f)   #用csv模块读取f
        headers=next(reader)    #用next的迭代遍历
        print(headers)          #读取csv的头行
        for row in reader:
            # print(root + row[0])
            if row[1] == '-1':
                shutil.copy(root + row[0], female + row[0] )
            else:
                shutil.copy(root + row[0], male + row[0] )






if __name__ == '__main__':
    csv_read()
