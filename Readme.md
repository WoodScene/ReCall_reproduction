
1. 配置环境
cd ReCall
pip3 install -e .

2. 启动Sandbox Serving
cd ReCall/scripts/serving
python sandbox.py --port 2501


3. 训练
cd ReCall/scripts/train

4卡训练配置
nohup bash train_ours2.sh \
    --train_batch_size 4 \
    --ppo_mini_batch_size 1 \
    --use_re_call True \
    --prompt_template_name re_call_template_sys \
    --actor_model_path /group/40059/yuujiefeng/Backbones/ReSearch-Qwen2.5-3B-Instruct2 \
    --search_url http://0.0.0.0:2502 \
    --sandbox_url http://0.0.0.0:2501 \
    --project_name yujie \
    --experiment_name ours4 \
    --nnodes 1 \
    --n_gpus_per_node 4 \
    --save_freq 500 \
    --test_freq 500 \
    --total_epochs 2 \
    --wandb_api_key None \
    --save_path ./save_dir_ours4_Recall-3B \
    --train_files "['/group/40059/yuujiefeng/aaai2026/ReCall/data/ReCall-data/syntool_re_call/train.parquet']" \
    --test_files "['/group/40059/yuujiefeng/aaai2026/ReCall/data/ReCall-data/syntool_re_call/test.parquet']" \
    --checkpoint_save ./checkpoints_ours4_Recall-3B \
    >> ./train_wandb_ours4_Recall3B.log 2>&1 &



4. 合并训练后的ckpt （注意需要copy一个config.json过去，以及vocab.json，还有其他的json配置文件）
python model_merger.py merge \
    --backend fsdp \
    --local_dir /group/40059/yuujiefeng/aaai2026/ReCall/scripts/train/save_dir_ours4/global_step_500/actor \
    --target_dir /group/40059/yuujiefeng/Backbones/Qwen2.5-3B-Instruct_ours4_global_step_500


5. 启动inference （注意这里的tp需要根据gpu的个数进行配置，虚拟环境是sglang）
python3 -m sglang.launch_server \
        --served-model-name ReSearch-Qwen2.5-3B-Instruct2 \
        --model-path /group/40059/yuujiefeng/Backbones/ReSearch-Qwen2.5-3B-Instruct2 \
        --tp 4 \
        --context-length 8192 \
        --enable-metrics \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port 30001 \
        --trust-remote-code \
        --disable-overlap-schedule \
        --disable-radix-cache

