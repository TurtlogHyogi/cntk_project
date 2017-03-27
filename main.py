import Dataset
import Train
import os

my_dataset_dir = r'D:\YSH_Github\cntk_project\datasets\images\in_data'
CIFAR_10 = r'D:\local\cntk\Examples\Image\DataSets\CIFAR-10'

out_dataset_dir = r'D:\YSH_Github\cntk_project\datasets\images\out_data'
out_model_dir = os.path.join(out_dataset_dir,'model')

if __name__ == '__main__':
#    Dataset.Dataset_create(my_dataset_dir,out_dataset_dir,32,32,framework=3)
#    print(Dataset.Dataset_result(out_dataset_dir))

    Train.Train_create(dataset_dir=out_dataset_dir, framework=3, out_model_dir=out_model_dir, max_epochs=80, mb_size=64, network_name='conv')
#    print(Train.Train_result(out_model_dir))