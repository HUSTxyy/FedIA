
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='Fed', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='MS', help='dataset name')#########
    parser.add_argument('--model', type=str,
                        default='UNet2D', help='model name')############
    parser.add_argument('--batch_size', type=int,
                        default=4, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=1e-4,
                        help='base learning rate')
    parser.add_argument('--bilinear', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    # for FL
    parser.add_argument('--n_clients', type=int,  default=4,
                        help='number of users') 
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=1, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=300, help='rounds')

    parser.add_argument('--warm', type=int,  default=1)


    parser.add_argument("--data_dir", type=str, default='/home/xyy/append/In3D/')
    parser.add_argument("--train_list0", type=str, default='c0.txt')
    parser.add_argument("--train_list1", type=str, default='c1.txt')
    parser.add_argument("--train_list2", type=str, default='c2.txt')
    parser.add_argument("--train_list3", type=str, default='c3.txt')
    
    parser.add_argument("--val_list0", type=str, default='c0test.txt')
    parser.add_argument("--val_list1", type=str, default='c1test.txt')
    parser.add_argument("--val_list2", type=str, default='c2test.txt')
    parser.add_argument("--val_list3", type=str, default='c3test.txt')


    
    args = parser.parse_args()
    return args