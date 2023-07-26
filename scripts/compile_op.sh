cd utils/nearest_neighbors
###
 # @Author: BigCiLeng && bigcileng@outlook.com
 # @Date: 2023-07-25 11:49:49
 # @LastEditors: BigCiLeng && bigcileng@outlook.com
 # @LastEditTime: 2023-07-25 11:51:45
 # @FilePath: /RandLA-Net-Pytorch-New/compile_op.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
### 
python setup.py install --home="."
cd ../../

cd utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../../