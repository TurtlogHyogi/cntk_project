import Dataset
import Train

my_dataset_dir = Dataset.my_dataset_dir # r'D:\Github\dataset\img\mydataset'

out_dataset_dir = Dataset.out_dataset_dir # r'D:\Github\dataset\img\mydataset\out_dataset'
out_model_dir = Train.out_model_dir

if __name__ == '__main__':
    Dataset.Dataset_create(my_dataset_dir,out_dataset_dir,32,32,framework=3)
    print(Dataset.Dataset_result(out_dataset_dir))
    Train.Train_create(dataset_dir=out_dataset_dir, framework=3, out_model_dir=out_model_dir, max_epochs=20, mb_size=300, network_name='conv')
    print(Train.Train_result(out_model_dir))
