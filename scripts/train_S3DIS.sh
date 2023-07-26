TRAIN_S3DIS(){
###
 # @Author: BigCiLeng && bigcileng@outlook.com
 # @Date: 2023-06-20 16:43:14
 # @LastEditors: BigCiLeng && bigcileng@outlook.com
 # @LastEditTime: 2023-07-26 01:11:54
 # @FilePath: /RandLA-Net-Pytorch-New/scripts/train_S3DIS.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
### 
cd ..
dataset_name='S3DIS'
work_type='train'

adam_lr=1e-2  # initial learning rate
scheduler_gamma=0.95

device='gpu'
device_nums=2
num_workers=4
python -W ignore randla_system.py --dataset_name $dataset_name --work_type $work_type \
                --adam_lr $adam_lr --scheduler_gamma $scheduler_gamma \
                --device $device --device_nums $device_nums --num_workers $num_workers
}
TRAIN_S3DIS