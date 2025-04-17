# ckpt_dir="/model_zoo/Wan2.1-I2V-14B-480P-Diffusers/"
# ckpt_dir="/model_zoo/Wan2.1-I2V-14B-480P"
# echo "python generate.py  --task t2v-14B --size 1280*720 --ckpt_dir $ckpt_dir --prompt 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.'"
# echo "python generate.py  --task i2v-14B --size 832*480 --ckpt_dir $ckpt_dir" --offload_model False


prompt=$(cat examples/prompt.txt)
image_path="examples/i2v_input.JPG"
no_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  

# model_id="/model_zoo/Wan2.1-I2V-14B-480P-Diffusers"
# python generate_diffusers.py 

model_id="/model_zoo/Wan2.1-I2V-14B-480P"
python generate.py \
    --task i2v-14B \
    --size 832*480 \
    --sample_steps 30 \
    --ckpt_dir $model_id \
    --image $image_path \
    --prompt "$prompt"\
    --base_seed 42 \
    --offload False # \
    # --compile

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  