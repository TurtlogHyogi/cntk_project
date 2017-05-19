from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
from math import log10
import xml.etree.ElementTree as et
import xml.dom.minidom
import threading,time
import logging

# set default dataset_args -> options for making Dataset
# options : path,encoding,log
# return : default dataset_args
#
# ex)   test = parse_args()
#       then test.root='/', test.out='/', test.log_start=False
#       then if you want change log_start, test.logstart=True
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
                                     description='Create dataset(resized_img.png, mean_fil.xml, map.txt, label.txt')
    parser.add_argument('--root', default='/', 
                        help='path of input dataset.')
    parser.add_argument('--out', default='/', 
                        help='path of output dataset.')

    dgroup = parser.add_argument_group('Options for creating Dataset')
    dgroup.add_argument('--resize', type=int, default=50, 
                        help='resize raw_img to resized_img(resize by resize)')
    dgroup.add_argument('--channel', type=int, default=3, 
                        help='channels of image. RGB -> 3, Gray -> 1')
    dgroup.add_argument('--total-img-num', type=int, default=0, 
                        help='logging data, total number of dataset')
    dgroup.add_argument('--current-img-num', type=int, default=0,
                        help='logging data, number of dataset made')
    dgroup.add_argument('--log-start', type=bool, default=False, 
                        help='if --log-start=True -> running print_log thread, else -> stop print_log thread')
    args = parser.parse_args()

    return args


# get foldernames
# in : abs_path-> return : foldernames, type(foldernames)==list
#
# ex)   /root/test_path/folder1/image0.bmp
#       /root/test_path/folder2/image1.png
#       /root/test_path/folder2/image2.bmp
#       /root/test_path/folder2/image3.jpeg
#       /root/test_path/folder3/image4.jpg
#       /root/test_path/folder3/image5.bmp
#       /root/test_path/folder3/image6.png
#
#       foldernames = list_get_foldernames('/root/test_path')
#       foldernames == [folder1,folder2,folder3]
def list_get_foldernames(in_dataset_dir):
    filenames = os.listdir(in_dataset_dir)    
    foldernames = []
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '':
            foldernames.append(filename)

    return foldernames
    #return list(folder for folder in os.listdir(in_dataset_dir) if os.path.isdir(os.path.join(in_dataset_dir,folder)))
    
# get imgnames
# in : abs_path -> return imgnames, type(imgnames)==list
#
# ex)   /root/test_path/folder1/image0.bmp
#       /root/test_path/folder2/image1.png
#       /root/test_path/folder2/image2.bmp
#       /root/test_path/folder2/image3.jpeg
#       /root/test_path/folder3/image4.jpg
#       /root/test_path/folder3/image5.bmp
#       /root/test_path/folder3/image6.png
#
#       imgnames = list_get_imgnames('/root/test_path/folder1') -> imgnames == [image0.bmp]
#       imgnames = list_get_imgnames('/root/test_path/folder2') -> imgnames == [image1.png, image2.bmp, image3.jpeg]
#       imgnames = list_get_imgnames('/root/test_path/folder3') -> imgnames == [image4.jpg, image5.bmp, image6.png]
def list_get_imgnames(foldername):
    imgnames = []
    imgnames = os.listdir(foldername)    

    return imgnames    
    #return list(imgname for imgname in os.listdir(foldername) if os.path.splitext(os.path.join(in_dataset_dir,imgnames))[1] in ['.jpg','.png','.jpeg','.bmp'])

# save  : raw_img -> resized_img(.png)
# in_filename : abs_path, out_filename : abs_path, resize : integer
# 
# ex)   /root/test_path/folder1/image0.bmp
#
#       save_resized_PNG_img('/root/test_path/folder1/image0.bmp','/out/test.png',100)
#       
#       /root/test_path/folder1/image0.bmp
#       /out/test.png
def save_resized_PNG_img(in_filename,out_filename,resize):
    raw_img = Image.open(in_filename)
    resized_img = raw_img.resize((resize,resize))
    resized_img.save(out_filename)               

# savemean(fname,data,dataset_args) -> make file : fname
# fname : file_name. ex) test.xml
# data : R_mean_array_data + G_mean_array + B_mean_array.
#        data looks like MeanImg
#
# ex) 3*2*2 image -> R_mean = [123.123123, 156.231523, 126.215664, 152.126667]
#                    G_mean = [111.125125, 185.235121, 192.123151, 132.115251]
#                    B_mean = [92.412415, 125.124125, 125.135234, 199.123125]
#                 -> data = [123.123123, 156.231523, 126.215664, 152.126667] + [111.125125, 185.235121, 192.123151, 132.115251] + [92.412415, 125.124125, 125.135234, 199.123125]
#                                                R_mean                                             G_mean                                            B_mean
#                         = [123.123123, 156.231423, 126.215664, 152.126667, 111.125125, 185.235121, 192.123151, 132.115251, 92.412415, 125.124125, 125.135234, 199.123125]
#
# then, write file. format of the file is xml.
def savemean(fname,data,dataset_args):
    root = et.Element('opencv_storage')
    et.SubElement(root,'Channel').text = str(dataset_args.channel)
    et.SubElement(root,'Row').text = str(dataset_args.resize)
    et.SubElement(root,'Col').text = str(dataset_args.resize)
    meanImg = et.SubElement(root,'MeanImg', type_id = 'opencv-matrix')
    et.SubElement(meanImg,'rows').text = '1'
    et.SubElement(meanImg,'cols').text = str(dataset_args.channel*dataset_args.resize*dataset_args.resize)
    et.SubElement(meanImg,'dt').text = 'f'

    et.SubElement(meanImg,'data').text = ' '.join('%e' % data[i] for i in range(dataset_args.channel*
                                                                                dataset_args.resize*
                                                                                dataset_args.resize))

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = ' '))
 
# save resized_img & write map_file, labels_file, mean_file
# map_file
#   title : 'train_map.txt' 
#   format : 'fname_abs_path' + ''\t' + 'label
#             fname_abs_path' + ''\t' + 'label
#             ...'
# labels_file
#   title : 'label.txt'
#   format : 'foldername1
#             foldername2
#             ...'
# mean_file
#   title : RGB_mean.xml
def create_dataset(in_dataset_dir, out_dataset_dir, resize, framework, dataset_args):
    if not (os.path.exists(in_dataset_dir) and os.listdir(in_dataset_dir)): # check input_dataset
        return print('Dataset is Wrong.')
    
    if not os.path.exists(out_dataset_dir):
        os.makedirs(out_dataset_dir)
    
    dataset_args.log_start = True
    with open(out_dataset_dir+'/train_map.txt','w') as map:
        with open(out_dataset_dir+'/labels.txt','w') as labels:
            # start thread(print_dataset_log)

            foldernames = list_get_foldernames(in_dataset_dir)
            label=0
            img_mean = np.zeros((resize,resize,3)) 

            # make dataset folder by folder
            for foldername in foldernames: 
                labels.write(foldername+'\n') # write label(foldername)
                abs_in_foldername = os.path.join(in_dataset_dir,foldername)
                abs_out_foldername = os.path.join(out_dataset_dir,'train_db') 
                if not os.path.exists(abs_out_foldername):
                    os.makedirs(abs_out_foldername)
                imgnames = list_get_imgnames(abs_in_foldername) # get imgnames in foldername
                
                for in_imgname in imgnames: 
                    abs_in_imgname = os.path.join(abs_in_foldername,in_imgname) 
                    extension = ['.jpg','.png','.jpeg','.bmp']
                    if os.path.splitext(abs_in_imgname)[1] in extension: # check extension(is it img_file?)
                        out_imgname = '{:0{}d}.png'.format(dataset_args.current_img_num,int(log10(dataset_args.total_img_num)+1)) # set out_image_name
                        abs_out_imgname = os.path.join(abs_out_foldername,out_imgname)
                        save_resized_PNG_img(abs_in_imgname,abs_out_imgname,resize) # save resized_png_img

                        img = Image.open(abs_out_imgname)
                        # check. is it RGB image?
                        if img.mode == 'RGB':
                            map.write(abs_out_imgname+'\t'+str(label)+'\n')
                            arr = np.array(img,dtype=np.float32)
                            img_mean += arr/dataset_args.total_img_num # calculate RGB_img_mean
                            dataset_args.current_img_num += 1
                label+=1

    img_mean = np.ascontiguousarray(np.transpose(img_mean,(2,0,1)))
    img_mean = img_mean.reshape(3*resize*resize)
    savemean(out_dataset_dir + '/train_mean.xml',img_mean, dataset_args)
    dataset_args.log_start = False # stop print_log thread
    
    return True

# print log
# you can chane logging.Formatter or message
def print_dataset_log(in_dataset_dir, out_dataset_dir, resize, framework, dataset_args):
    # wait until train_map.txt & labels.txt made
    while not dataset_args.log_start:
        pass

    # set logger
    logger = logging.getLogger('Dataset')
    logger.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(os.path.join(out_dataset_dir,'create_val_db.log'),'w')
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s:%(asctime)s], %(message)s')
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    
    # print_log
    while dataset_args.log_start == True:
        message = 'Total={0}, Current={1}, Progress={2:0.4f}%'.format(dataset_args.total_img_num,
                                                                     dataset_args.current_img_num,
                                                                     dataset_args.current_img_num*100/dataset_args.total_img_num)
        logger.info(message)
        time.sleep(1)
    if dataset_args.log_start == False:
        message = 'Total={0}, Current={1}, Progress={2:0.4f}%'.format(dataset_args.total_img_num,
                                                                     dataset_args.current_img_num,
                                                                     dataset_args.current_img_num*100/dataset_args.total_img_num)
        logger.info(message)
        

def Dataset_result(out_dataset_dir):
    found_dir = os.path.exists(out_dataset_dir)
    found_files = os.listdir(out_dataset_dir)
    found_dataset = []

    if not found_dir:
        return print('Directory is not found')    

    if not found_files:
        return print('Dataset is not found')
    
    for found_file in found_files:
        if os.path.splitext(found_file)[1] in ('.txt','.xml'):
            found_file = os.path.join(out_dataset_dir,found_file)
            found_dataset.append(found_file)

    found_dataset.sort()

    return found_dataset
    
def Dataset_create(in_dataset_dir, out_dataset_dir, resize, framework):
    if framework == 3: # CNTK
        # set default dataset_args
        dataset_args = parse_args()
        dataset_args.root = in_dataset_dir
        dataset_args.out = out_dataset_dir
        dataset_args.resize = resize
    
        # count total_img_num
        foldernames = list_get_foldernames(in_dataset_dir)
        for foldername in foldernames:
            abs_in_foldername = os.path.join(in_dataset_dir,foldername)
            imgnames = list_get_imgnames(abs_in_foldername)
        
            extension = ['.jpg','.png','.jpeg','.bmp']
            for imgname in imgnames:
                if os.path.splitext(imgname)[1] in extension:
                    dataset_args.total_img_num += 1

        # run threads
        functions = [create_dataset,print_dataset_log]
        func_args = (in_dataset_dir,out_dataset_dir,resize,framework,dataset_args)
        threads = []
        for function in functions:
            th = threading.Thread(target=function, args=func_args, name = function)
            th.start()
            threads.append(th)
        for thread in threads:
            thread.join()
        print('Dataset_create finish')

    return True

w_my_dataset_dir = r'D:\Github\cntk_dataset\img'
w_out_dataset_dir = r'D:\Github\cntk_dataset\outdataset'
l_my_dataset_dir = r'/root/git/cntk_dataset/img'
l_out_dataset_dir = r'/root/git/cntk_dataset/out_dataset'

if __name__ == '__main__':
    Dataset_create(in_dataset_dir = w_my_dataset_dir,
                   out_dataset_dir = w_out_dataset_dir,
                   resize = 32,
                   framework = 3)
    print(Dataset_result(w_out_dataset_dir))

