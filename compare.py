"""
pip3 install imagededup
使用CNN计算图像间的相似度,进行图像清洗.
--root_dir 输入根路径
--save_path 输出根路径
--start_num 起始计数.
--
"""
import argparse
from logging import root
import cv2
from imagededup.methods import CNN
import os
from shutil import copyfile

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("-dir", default="label",type=str, help="输入根路径.")
parser.add_argument("-out", type=str,default="output", help="输出根路径.")
parser.add_argument("-t", type=float,default=0.99, help="相似度阈值.")
arg = parser.parse_args()

def main():
	default_img_format = [".jpg",".png",".jpeg"]
	img_list = []
	phasher = CNN()
	src_dir = arg.dir
	out_dir = arg.out
	thres = arg.t

	#Generate img list to travel
	for root, dirs, files in os.walk(src_dir, topdown=False):
		for file_name in files:
			# Continue while the file is illegal.
			if (file_name.endswith(extend) for extend in default_img_format):
				img_list.append(file_name)
			else:
				continue
	#Generateing CNN encoding
 
	print('='*20)
	print("Image list generated,generating CNN encodings...")
	print('='*20)
	encodings = phasher.encode_images(src_dir)
	duplicates = phasher.find_duplicates_to_remove(encoding_map=encodings, min_similarity_threshold=thres)
	# print(duplicates)

	#Generating Final List
	print('='*20)
	print("Find duplicates done,generating Final list...")
	print('='*20)
	img_to_copy = img_list.copy()
	for img_to_remove in duplicates:
		img_to_copy.remove(img_to_remove)

	#Copying accepted images
	print('='*20)
	print("Final list generated,Copying accepted images...")
	print('='*20)
	for img_name in img_to_copy:
		copyfile(os.path.join(src_dir, img_name), os.path.join(out_dir, img_name))

	print('='*20)
	print("Done...")
	print("Accepted {} images successfully.".format(str(len(img_to_copy))))
	print('='*20)

if __name__ == "__main__":
	main()
