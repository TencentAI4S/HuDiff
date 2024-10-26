export CUDA_VISIBLE_DEVICES=2
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
UNPAIR_DATA_PATH='oas_heavy_human_data/heavy_nano_idx.pt'
CONFIG_PATH='configs/heavy_train.yml'
LOG_PATH='tmp/heavy_train_log/'
python nanobody_scripts/nanotrain.py \
--unpair_data_path $UNPAIR_DATA_PATH \
--config_path $CONFIG_PATH \
--log_path $LOG_PATH \