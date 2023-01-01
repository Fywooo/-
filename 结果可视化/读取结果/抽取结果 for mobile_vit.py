str = []


with open(r"C:/Users/lenovo/Desktop/实验结果/mobile_Vit + 60 epoch/新建文本文档.txt", "r",encoding="utf-8-sig") as f:  # 打开文件
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        if 'acc' in line and 'train' in line:
            print(line)
            # for epoch_num
            # str.append( line[6] + line[7] )
            # print( "'" + line[6] + line[7] + "'")
            # print(line[14 : 20])


            # for loss:
            # if(',' in line[23 : 28]):
            #     str.append(float(line[22 : 27]))
            # else:
            #     str.append(float(line[23 : 28]))
            # print(line[23 : 28])

            # # for acc:
            # if(':' in line[35 : 40]):
            #     str.append(float(line[34 : 39]))
            # else:
            #     str.append(float(line[35 : 40]))
            # print(line[35 : 40])


# print(str)