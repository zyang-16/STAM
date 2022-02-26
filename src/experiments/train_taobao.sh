export CUDA_VISIBLE_DEVICES=7
python main.py --dataset taobao --input_dim 64 --num_layers 3 --n_heads 4 --hidden_dim 64 --dim 128 --maxlen 20 --hidden_dim 64 --print_step 500 --lr 0.001 --batch_size 2048 --epoch 50
