export CUDA_VISIBLE_DEVICES=5
python main.py --dataset ml-1m --input_dim 64 --hidden_dim 64 --dim 128 --num_layers 3 --n_heads 4 --maxlen 200  --lr 0.001 --batch_size 1024 --epoch 300
