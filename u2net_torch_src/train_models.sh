#### @Chao Huang(huangchao09@zju.edu.cn).
############################
#### independent models ####
############################
CUDA_VISIBLE_DEVICES=0 python train_model_no_adapters.py --trainMode 'independent' --out_tag '20190312' --tasks 'Task02_Heart'
CUDA_VISIBLE_DEVICES=0 python train_model_no_adapters.py --trainMode 'independent' --out_tag '20190312' --tasks 'Task03_Liver'
CUDA_VISIBLE_DEVICES=0 python train_model_no_adapters.py --trainMode 'independent' --out_tag '20190312' --tasks 'Task04_Hippocampus'
CUDA_VISIBLE_DEVICES=0 python train_model_no_adapters.py --trainMode 'independent' --out_tag '20190312' --tasks 'Task05_Prostate'
CUDA_VISIBLE_DEVICES=0 python train_model_no_adapters.py --trainMode 'independent' --out_tag '20190312' --tasks 'Task07_Pancreas'
CUDA_VISIBLE_DEVICES=0 python train_model_no_adapters.py --trainMode 'independent' --out_tag '20190312' --tasks 'Task09_Spleen'




############################
####### shared model #######
############################
CUDA_VISIBLE_DEVICES=0 python train_model_wt_adapters.py --trainMode 'shared' --out_tag '20190312' --tasks 'Task02_Heart' 'Task03_Liver' 'Task04_Hippocampus' 'Task05_Prostate' 'Task07_Pancreas'

## adapted to Task09_Spleen
# CUDA_VISIBLE_DEVICES=0 python train_model_wt_adapters.py --trainMode 'shared' --out_tag '20190312_adapted2spleen' --tasks 'Task09_Spleen' --ckp 'xx/xx/xx.pth.tar'




############################
##### universal model #####
############################
CUDA_VISIBLE_DEVICES=0 python train_model_wt_adapters.py --trainMode 'universal' --module 'separable_adapter' --base_outChans 16 --out_tag '20190312' --tasks 'Task02_Heart' 'Task03_Liver' 'Task04_Hippocampus' 'Task05_Prostate' 'Task07_Pancreas'
## adapted to Task09_Spleen.
# CUDA_VISIBLE_DEVICES=0 python train_model_wt_adapters.py --trainMode 'universal' --module 'separable_adapter' --base_outChans 16 --out_tag '20190312_sep_adapted2spleen' --tasks 'Task09_Spleen' --ckp 'xx/xx/xx.pth.tar'









