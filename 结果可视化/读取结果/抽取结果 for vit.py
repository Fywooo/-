str = []


with open(r"C:/Users/lenovo/Desktop/神经网络课设/卷积神经网络与ViT结合的探索与实现/实验结果/纯VIT + 60 epoch/新建文本文档.txt", "r",encoding="utf-8-sig") as f:  # 打开文件
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        if 'Precission' in line:
            print(line)
            # for epoch_num
            # str.append( line[6] + line[7] )
            # print( "'" + line[6] + line[7] + "'")
            # print(line[14 : 20])


            # # for loss:
            # if(',' in line[14 : 20]):
            #     str.append(float(line[13 : 19]))
            # else:
            #     str.append(float(line[14 : 20]))

            # # for acc:
            if(',' in line[32 : 38]):
                str.append(float(line[31 : 37]))
            else:
                str.append(float(line[32 : 38]))
            # print(line[32 : 38])


print(str)

