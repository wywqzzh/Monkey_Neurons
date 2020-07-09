def CoherenceSelect():
    f = open("../data/Coherence.txt", "r")  # 设置文件对象
    line = f.readline()
    line = line[:-1]
    collection=[]
    while line:  # 直到读取完文件
        line = f.readline()  # 读取一行文件，包括换行符
        if len(line)!=0:
            x=line.split('\t')
            Coherence=int(x[1].strip('\n'))
            if Coherence==100:
                collection.append(x[0][:-4])
    # print(collection)
    # print(len(collection))
    f.close()  # 关闭文件
    return collection