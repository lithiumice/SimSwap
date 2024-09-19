
# Use Ray to Accelerate

    cd /home/hualin/code/DeepFace/SimSwap
    conda activate simswap2

    python test_video_swapmulti_ray.py \
        --crop_size 224 \
        --use_mask \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --pic_a_path debug/target.jpg \
        --video_path debug/rank5.mp4 \
        --output_path ./output/s1/rank5_swap_224_ray.mp4 \
        --temp_path ./debug/temp_results/rank5_swap_224  \
        --no_simswaplogo 
        
    python test_video_swapmulti_ray.py \
        --crop_size 224 \
        --use_mask \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --pic_a_path debug/target.jpg \
        --video_path debug/rank6.mp4 \
        --output_path ./output/s1/rank6.mp4 \
        --temp_path ./debug/temp_results/rank6  \
        --no_simswaplogo 
        

