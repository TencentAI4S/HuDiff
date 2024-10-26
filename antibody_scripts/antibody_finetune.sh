export CUDA_VISIBLE_DEVICES=3
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
PAIR_DATA_PATH='data/release_data_dir/oas_pair_mouse_data'
CONFIG_PATH='configs/antibody_finetune.yml'
LOG_PATH='tmp/antibody_finetune_log/'
CKPT_PATH='antibody/pretrain_antibody.pt'
DATA_VERSION='filter'
export CONSIDER_MOUSE=True
python antibody_scripts/antibody_finetune.py \
--pair_mouse_data_path $PAIR_DATA_PATH \
--config_path $CONFIG_PATH \
--log_path $LOG_PATH \
--ckpt_path $CKPT_PATH \
--data_version $DATA_VERSION \
--consider_mouse $CONSIDER_MOUSE \