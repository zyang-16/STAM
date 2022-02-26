import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import placeholder 
from load_data import Data
from parser import parse_args
from model import STAM4Rec
import time
from evaluate import evaluate  
import os
import faiss
import time 

def train(data_generator, args):
    train_data = data_generator.train_data

    config = dict() 
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['temporal_table'] = data_generator.train_temporal_table
    norm_adj = data_generator.get_adj_mat() 
    config['norm_adj'] = norm_adj
    print("temporal_table", config['temporal_table'])
    #np.savetxt('temporal.txt', config['temporal_table'], fmt="%d", delimiter=" ")
    
    """
    ********************************************
    load model
    """
    placeholders = {
        'users':tf.compat.v1.placeholder(tf.int32, shape=(None), name="users"),
        'pos_items':tf.compat.v1.placeholder(tf.int32, shape=(None), name="pos_items"),
        'neg_items':tf.compat.v1.placeholder(tf.int32, shape=(None), name="neg_items"),
    }
    model = STAM4Rec(placeholders, config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer()) 
    tensorboard_model_path = './tensorboards/'
    
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    summary_write = tf.summary.FileWriter(tensorboard_model_path, sess.graph)   

    """
    *********************************************************
    Train.
    """
    mrr_loger, ndcg_loger, hit_loger = [], [], []  
    total_steps = 0
    for epoch in range(1, args.epochs+1):
        batch_data = data_generator.get_batch_data(train_data)
        batch_num = len(batch_data)
        # t1 = time.time()
        stam_weights = sess.run(model.stam_weights)
        # print("t1....", time.time() - t1)
        for idx in range(batch_num):
            total_steps += 1
            users, pos_items, neg_items = data_generator.sample(batch_data[idx])
            train_time = time.time() 
            _, batch_loss = sess.run([model.opt_op, model.bpr_loss], feed_dict={model.users:users, model.pos_items:pos_items, model.neg_items:neg_items, model.A_new:stam_weights})
        
        

            if total_steps % args.print_step == 0: 
                    # summary_write.add_summary(sess.run(merged), total_steps)
                    print("Idx:", '%04d' % idx,
                        "train_loss=", "{:.5f}".format(batch_loss),
                        "time=", "{:.5f}".format(time.time() - train_time)) 
        """  
        *********************************************************
        Test.
        """
        ret = test(sess, model, data_generator, norm_adj, args)
        perf_str = 'Epoch %d: mrr=[%s], ndcg=[%s], hr=[%s]' % \
                   (epoch,    
                    ', '.join(['%.5f' % r for r in ret['mrr']]),
                    ', '.join(['%.5f' % r for r in ret['ndcg']]),
                    ', '.join(['%.5f' % r for r in ret['hits']])) 
        print(perf_str) 
        mrr_loger.append(ret['mrr'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hits']) 

    mrrs = np.array(mrr_loger)
    ndcgs = np.array(ndcg_loger) 
    hrs = np.array(hit_loger)

    best_mrr_0 = max(mrrs[:, 0])
    idx = list(mrrs[:, 0]).index(best_mrr_0) 

    final_perf = "Best mrr=[%s], ndcg=[%s], hr=[%s]" % \
                 ('\t'.join(['%.5f' % r for r in mrrs[idx]]), 
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                  '\t'.join(['%.5f' % r for r in hrs[idx]])) 
    print(final_perf) 

def test(sess, model, data_generator, norm_adj, args): 
    test_users = list(data_generator.test_dict.keys()) 
    # print("test_users", test_users)
    Ks = eval(args.Ks)
    result = {'mrr': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)), 'hits': np.zeros(len(Ks))}

    u_batch_size = args.batch_size

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    count = 0
    item_batch = list(range(data_generator.n_users, data_generator.n_nodes)) # M ~ (M+N-1)
    # get item_embeds
    item_batch_size = args.batch_size
    n_item_batchs = len(item_batch) // item_batch_size + 1
    stam_weights = sess.run(model.stam_weights)
    for v_batch_id in range(n_item_batchs):
        start = v_batch_id * item_batch_size
        end = (v_batch_id + 1) * item_batch_size
        item_N = item_batch[start:end]
        item_embeds_batch = sess.run(model.pos_item_embed,{model.pos_items: item_N, model.A_new:stam_weights})
        if v_batch_id == 0:
            item_embeds = item_embeds_batch
        else:
            item_embeds = np.vstack((item_embeds, item_embeds_batch))
    #print("item_embeds", item_embeds.shape)
    
    d = item_embeds.shape[-1]

    index = faiss.IndexFlatIP(d) 
    index.add(item_embeds)

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        user_embeds = sess.run(model.user_emb, {model.users: user_batch, model.A_new:stam_weights})

        D, I = index.search(user_embeds, 200) 

        recommended_items = I 
        rate_batch = np.array(recommended_items)
        
        test_items = []
        for user in user_batch:
            test_items.append(data_generator.test_dict[user])# (B, #test_items) # item idx 0~(N-1)
        for idx, user in enumerate(user_batch):
            train_items_off = data_generator.train_dict[user]
            rate_batch[idx] = [-1 if p in train_items_off else p for p in rate_batch[idx]]
        
        batch_result = evaluate(rate_batch, test_items, Ks)#(B,k*metric_num) 
        count += len(batch_result)
        
        for re in batch_result:
            result['mrr'] += re['mrr'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hits'] += re['hits'] / n_test_users
            
    return result

if __name__ == "__main__":
    args = parse_args()
    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, maxlen=args.maxlen)
    train(data_generator, args)
