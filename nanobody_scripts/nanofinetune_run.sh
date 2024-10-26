export CUDA_VISIBLE_DEVICES=1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
VHH_DATA_PATH='oas_vhh_data/vhh_nano_idx.pt' 
CONFIG_PATH='configs/training_nano_framework.yml' 
LOG_PATH='tmp/nano_finetune_log/' 
python nanobody_scripts/nanofinetune.py \
--vhh_data_fpath $VHH_DATA_PATH \
--config_path $CONFIG_PATH \
--log_path $LOG_PATH \