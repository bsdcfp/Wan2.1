
# model_id="/model_zoo/Wan2.1-I2V-14B-480P-Diffusers"
model_id="/model_zoo/Wan2.1-T2V-1.3B"

prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  

python generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --frame_num 81 \
    --ckpt_dir $model_id \
    --offload False \
    --prompt "$prompt"

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  