#exp='0512-mage'
#CUDA_VISIBLE_DEVICES=0 python task_spec.py --task age  --lr 1e-3 --train shhs1 --valid shhs2 --input eeg-mage --bs 128 --exp $exp --weight_deep 2 --model_mage 20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.0 &
#CUDA_VISIBLE_DEVICES=1 python task_spec.py --task stage  --lr 3e-3 --train shhs1 --valid shhs2 --input eeg-mage --bs 128 --exp $exp --weight_deep 2 --model_mage 20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.0 &
#CUDA_VISIBLE_DEVICES=2 python task_spec.py --task age  --lr 1e-3 --train shhs1 --valid shhs2 --input eeg-mage --bs 128 --exp $exp --weight_deep 2 --model_mage 20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.5 &
#CUDA_VISIBLE_DEVICES=3 python task_spec.py --task stage  --lr 3e-3 --train shhs1 --valid shhs2 --input eeg-mage --bs 128 --exp $exp --weight_deep 2 --model_mage 20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.5 &

# using multitaper EEG spec to do age prediction
exp='for ali'
CUDA_VISIBLE_DEVICES=0 python task_spec.py --task age  --lr 1e-3 --train shhs1 --valid shhs2 --input eeg-mt --bs 128 --exp $exp
