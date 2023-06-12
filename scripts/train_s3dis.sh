TRAIN_SEMANTIC3D(){
pretrain_name=TRAIN_SEMANTIC3D
cd ..
dataset_name='S3DIS'
dataset='/share/dataset/sqn_own/S3DIS'
num_workers=2

num_classes=13
num_points=40960  # Number of input points
sub_grid_size=0.04  # preprocess_parameter
dataset_sampling='active_learning'
decimation=4
neighbors=16

batch_size=2  # batch_size during training
train_steps=500  # Number of steps per epochs
val_steps=100  # Number of validation steps per epoch
adam_lr=1e-2  # initial learning rate
scheduler_gamma=0.95
epochs=100

work_type='train'
device='gpu'
gpu=2

python -W ignore randla_system.py --dataset_name $dataset_name --dataset $dataset \
                --work_type $work_type --num_classes $num_classes --epochs $epochs \
                --num_points $num_points --sub_grid_size $sub_grid_size --train_steps $train_steps \
                --val_steps $val_steps --adam_lr $adam_lr --batch_size $batch_size \
                --decimation $decimation --dataset_sampling $dataset_sampling --neighbors $neighbors \
                --scheduler_gamma $scheduler_gamma --device $device --gpu $gpu --num_workers $num_workers

}

TRAIN_SEMANTIC3D