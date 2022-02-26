import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CogRec Model")
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='input data path')
    parser.add_argument('--name', type=str, default='STAM',
                        help='input data path')
    parser.add_argument('--dataset', type=str, default='ml-1m/',
                        help='choose dataset for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--input_dim', type=int, default=64,
                        help='input dim(embedding size)') 
    parser.add_argument('--decay', type=int, default=1e-5,
                        help='decay for regularizer') 
    parser.add_argument('--num_layers', type=int, default=2,
                        help='the layer sizes of propagations') 
    parser.add_argument('--maxlen', type=int, default=200,
                        help='the length of input sequence')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='the number of attention heads')  
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden dim for STAM') 
    parser.add_argument('--dim', type=int, default=128,
                        help='dim for FFN') 
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--print_step', type=int, default=100,
                        help='how often to print training info.')
    parser.add_argument('--user_num', type=int, default=0,
                        help='number of users')
    parser.add_argument('--item_num', type=int, default=0,
                        help='number of items.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--attn_drop', type=float, default=0.5,
                        help='attn drop')
    parser.add_argument('--max_gradient_norm', type=float, default=1.0,
                        help='Clip gradients to this norm')
    parser.add_argument('--Ks', nargs='?', default='[20, 50]',
                        help='Output sizes of every layer')
    return parser.parse_args()

