model_name_or_path=meta-llama/Llama-3.1-8B

wbits=4
groupsize=128
output_dir=${model_name_or_path}_q${wbits}_g${groupsize}_c4
mkdir -p $output_dir



CUDA_VISIBLE_DEVICES=0 python model/llama.py \
    meta-llama/Llama-3.1-8B \
    --wbits $wbits \
    --groupsize $groupsize \
    --dataset c4 \
    --lat \
    --bcq_round 20 \
    --save $output_dir
    # --save $save_dir    > $save_dir/log.txt

# CUDA_VISIBLE_DEVICES=0 python lmeval.py \
#     --model hf \
#     --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda > $output_dir/wiki.txt
