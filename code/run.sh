for i in 1 2 3 4 5
do
 CUDA_VISIBLE_DEVICES=0 \
 python3 main.py \
 --input ../data/OpenEA/D_W_15K_V1/721_5fold/$i/ \
   --gcn RGC \
   --embedding_module TrainerTorch \
   --rel_param 0.0 \
   --rel_align_param 10 \
   --threshold 0.5 \
   --learning_rate 0.001 \
   --start_valid 25 \
   --eval_freq 25 \
   --neg_multi 50 \
   --neg_margin 1.5 \
   --neg_param 0.1 \
   --truncated_epsilon 0.98 \
   --batch_size 256 \
   --inverse_relation lin \
   --model_name d_w_best
done
