import os
#读取文件所在的文件夹
path="/home/dishierweidu/Desktop/temp/output"
files=os.listdir(path)  #读取文件夹下所有文件的路径
for i,name in enumerate(files):  #逐个遍历
    if name.find("txt")>=0 :    #判断文件名称中是否包含txt字符
        print(i)  #输出文件的个数    
        os.remove(path+"/"+name)  #删除文件   
