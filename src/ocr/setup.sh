pip3 install numpy scipy matplotlib pillow
pip3 install easydict opencv-python keras h5py PyYAML
pip3 install cython==0.24

# for gpu
pip3 install tensorflow-gpu==1.3.0
chmod +x ./ctpn/lib/utils/make.sh
cd ./ctpn/lib/utils/ && ./make.sh

# for cpu
# pip install tensorflow==1.3.0
# chmod +x ./ctpn/lib/utils/make_cpu.sh
# cd ./ctpn/lib/utils/ && ./make_cpu.sh
