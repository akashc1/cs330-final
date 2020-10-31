from protonet import run_protonet
from maml import run_maml
from argparse import ArgumentParser


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("learner", default="maml", choices=["maml", "protonet"],
                        help="Type of learning algorithm to use. Options are MAML and ProtoNet")
    parser.add_argument("--data-dir", default="../omniglot_resized")
    parser.add_argument("--log-dir", default="../logs/")

    parser.add_argument("--n-way", default=5, type=int)
    parser.add_argument("--k-shot", default=1, type=int)

    # MAML-specific args
    parser.add_argument("--learn-inner-update-lr", default=False, type=bool)
    parser.add_argument("--inner-update-lr", default=0.4, type=float)
    parser.add_argument("--num-inner-updates", default=1, type=int)
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument("--meta-batch-size", default=25, type=int)
    parser.add_argument("--num-filters", default=32, type=int)
    parser.add_argument("--meta-train", default=True, type=bool)
    parser.add_argument("--meta-train-iterations", default=15000, type=int)
    parser.add_argument("--meta-train-k-shot", default=-1, type=int)
    parser.add_argument("--meta-train-inner-update-lr", default=-1.0, type=float)
    
    # ProtoNet-specific args
    parser.add_argument("--n-query", default=5, type=int)
    parser.add_argument("--n-meta-test-way", default=5, type=int)
    parser.add_argument("--k-meta-test-shot", default=5, type=int)
    parser.add_argument("--n-meta-test-query", default=5, type=int)
    
    return parser

def main(args):

    if args.learner == "maml":
        run_maml(n_way=args.n_way,
                 k_shot=args.k_shot,
                 meta_batch_size=args.meta_batch_size,
                 meta_lr=args.meta_lr,
                 inner_update_lr=args.inner_update_lr,
                 num_filters=args.num_filters,
                 num_inner_updates=args.num_inner_updates,
                 learn_inner_update_lr=args.learn_inner_update_lr,
                 logdir=args.log_dir,
                 data_path=args.data_dir,
                 meta_train=args.meta_train,
                 meta_train_iterations=args.meta_train_iterations,
                 meta_train_k_shot=args.meta_train_k_shot,
                 meta_train_inner_update_lr=args.meta_train_inner_update_lr)
    elif args.learner == "protonet":
        run_protonet(data_path=args.data_dir,
                     n_way=args.n_way,
                     k_shot=args.k_shot,
                     n_query=args.n_query,
                     n_meta_test_way=args.n_meta_test_way,
                     k_meta_test_shot=args.k_meta_test_shot,
                     n_meta_test_query=args.n_meta_test_query,
                     logdir=args.log_dir)
    else:
        print("Invalid learner type!")

if __name__ == "__main__":
    parser = parse_args()
    ARGS = parser.parse_args()
    main(ARGS)

