import os
import time
import argparse
import torch


def get_options(args=None):

    parser = argparse.ArgumentParser(description="CMAES_PPO")

    # CMAES (basic_env) settings
    parser.add_argument('--backbone', default='cmaes', choices=['cmaes'], help='backbone algorithm')
    parser.add_argument('--m', type=int, default=10, help='number of subgroups')
    parser.add_argument('--subspace_dim', type=int, default=100, help='dimensionality of the subspace')
    parser.add_argument('--sigma', type=float, default=2, help='sigma for cmaes')
    parser.add_argument('--sub_popsize', type=int, default=20, help='population size of each subgroup')
    parser.add_argument('--max_fes', type=int, default=2.6e6, help='maximum number of function evaluations')
    parser.add_argument('--subFEs', type=int, default=1000, help='minimum number of iterations for each subgroup')
    parser.add_argument('--output_init_cma_info', type=bool, default=False, help='output the initial cma info')
    parser.add_argument('--initFEs', type=int, default=1000, help='number of iterations for initialization')

    # If add one more action, please change the action_space to 4
    parser.add_argument('--action_space', type=int, default=3, help='number of actions')

    #problem settings
    parser.add_argument('--divide_method', default="CEC2013LSGO", choices=["BNS","OB_nondep"], help='method to divide the problem set')

    #action settings
    parser.add_argument('--resource_list', type=list, default=[50000,100000,200000], help='list of resource limits')

    #rollout settings
    parser.add_argument('--one_problem_batch_size', type=int, default=4, help='number of instances for each problem')
    parser.add_argument('--per_eval_time',  type=int, default=1, help='number of evaluations for each instance')

    #PPO settings
    parser.add_argument('--each_question_batch_num', type=int, default=4, help='number of instances for each problem')

    # learning rate settings
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="learning rate for the critic network")
    parser.add_argument('--lr_model', type=float, default=1e-4, help="learning rate for the actor network")
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay per epoch')

    parser.add_argument('--state', default=[0.0 for _ in range(16)], help='initial state of actor')

    parser.add_argument('--test', type=int, default=0, help='swith to test mode')
    parser.add_argument('--device', default='cuda', choices=['cpu'], help='device to use for training / testing')

    parser.add_argument('--feature_num_1', type=int, default=16, help='number of features of each instance') # for actor1
    parser.add_argument('--feature_num_2', type=int, default=15, help='number of features of each instance') # for actor2
    parser.add_argument('--max_learning_step', type=int, default=10000, help='number of iterations for training')
    parser.add_argument('--update_best_model_epochs', type=int, default=1, help='number of iterations for training')

    #run settings
    parser.add_argument('--no_tb', action='store_true', help='disable Tensorboard logging')
    parser.add_argument('--no_saving', action='store_true', help='disable saving checkpoints')
    parser.add_argument('--log_dir', default='WCC/save_dir/log/', help='directory to write TensorBoard information to')
    parser.add_argument('--output_modal_dir', default='WCC/save_dir/ppo_model/', help='directory to write output models to')
    parser.add_argument('--output_data_dir', default=f'WCC/save_dir/running_data/', help='directory to write output data to')
    parser.add_argument('--output_rollout_dir', default='WCC/save_dir/rollout_data/', help='directory to write rollout output to')
    parser.add_argument('--test_dir', default='WCC/save_dir/test/', help='directory to write test output to')
    parser.add_argument('--run_name', default='model', help='name to identify the run')
    parser.add_argument('--RL_agent', default='ppo', choices=['ppo'], help='RL Training algorithm')
    parser.add_argument('--eval_only', action='store_true', default=False, help='switch to inference mode')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use')

    # parameters in framework
    parser.add_argument('--no_cuda', action='store_true', help='disable GPUs')
    parser.add_argument('--show_figs', action='store_true', help='enable figure logging')

    parser.add_argument('--use_assert', action='store_true', help='enable assertion')
    parser.add_argument('--no_DDP', action='store_true', help='disable distributed parallel')

    # Net(Attention Aggregation) parameters
    parser.add_argument('--v_range', type=float, default=6., help='to control the entropy')
    parser.add_argument('--encoder_head_num', type=int, default=4, help='head number of encoder')
    parser.add_argument('--decoder_head_num', type=int, default=4, help='head number of decoder')
    parser.add_argument('--critic_head_num', type=int, default=6, help='head number of critic encoder')

    parser.add_argument('--actor1_embedding_dim', type=int, default=128, help='dimension of input embeddings')
    parser.add_argument('--actor1_hidden_dim', type=int, default=256, help='dimension of hidden layers in Enc/Dec')
    parser.add_argument('--actor1_action_space', type=int, default=1, help='number of actions')

    parser.add_argument('--actor2_embedding_dim', type=int, default=160, help='dimension of input embeddings')
    parser.add_argument('--actor2_hidden_dim', type=int, default=320, help='dimension of hidden layers in Enc/Dec')
    parser.add_argument('--actor2_action_space', type=int, default=3, help='number of actions')

    parser.add_argument('--n_encode_layers', type=int, default=1, help='number of stacked layers in the encoder')
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")

    # Training parameters
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor for future rewards')
    parser.add_argument('--decision_interval', type=int, default=1, help='make action decision per decision_interval generations')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--T_train', type=int, default=2000, help='number of iterations for training')
    parser.add_argument('--n_step', type=int, default=5, help='n_step for return estimation')

    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--epoch_end', type=int, default=200, help='maximum training epoch')
    parser.add_argument('--epoch_size', type=int, default=200, help='number of instances per epoch during training')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='maximum L2 norm for gradient clipping')

    # Inference and validation parameters
    parser.add_argument('--Max_Eval', type=int, default=200000, help='number of obj evaluation for inference')
    parser.add_argument('--val_size', type=int, default=1024, help='number of instances for validation/inference')
    parser.add_argument('--greedy_rollout', action='store_true')
    parser.add_argument('--dataset_path', default=None,)
    parser.add_argument('--inference_interval', type=int, default=3)
    parser.add_argument('--load_path_for_test',
                        # 未更换网络前的模型 ： '/home/qiuwenjie/WCC/save_dir/ppo_model/model_CEC2013LSGO_1.0e+06_10_20_0.0001_20250405T090209/epoch-12.pt'
                        # 使用过的评估模型：'/home/qiuwenjie/WCC/save_dir/ppo_model/model_CEC2013LSGO_1.0e+06_10_20_0.0001_20250828T223947/epoch-4.pt'
                        # 评估模型，但效果不佳：'/home/qiuwenjie/WCC/save_dir/ppo_model/model_CEC2013LSGO_1.0e+06_10_20_0.0001_20250829T164119/epoch-90.pt'
                        default='/home/qiuwenjie/WCC/save_dir/ppo_model/model_CEC2013LSGO_1.0e+06_10_20_0.0001_20250829T164119/epoch-90.pt',
                        help='path to load model parameters for test')


    # resume and load models
    parser.add_argument('--load_path',
                        default=None,
                        help='path to load model parameters and optimizer state from')

    parser.add_argument('--resume',
                        default=None,
                        help='resume from previous checkpoint file')

    # logs/output settings
    parser.add_argument('--no_progress_bar', action='store_true', help='disable progress bar')

    parser.add_argument('--log_step', type=int, default=1, help='log info every log_step gradient steps')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    opts = parser.parse_args(args)

    opts.ns = opts.max_fes // opts.subFEs

    # figure out whether to use distributed training if needed
    opts.world_size = 1
    opts.distributed = False
    # opts.world_size = torch.cuda.device_count()
    # opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'
    # processing settings
    opts.use_cuda =  1

    opts.run_name = "{}_{}_{}_{}_{}_{}_{}".format(opts.run_name, opts.divide_method, "{:.1e}".format(opts.max_fes), opts.m, opts.sub_popsize, opts.lr_model, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]

    opts.modal_save_dir = os.path.join(
        opts.output_modal_dir,
        opts.run_name
    ) if not opts.no_saving else None

    opts.data_save_dir = os.path.join(
        opts.output_data_dir,
        opts.run_name
    ) if not opts.no_saving else None

    opts.rollout_save_dir = os.path.join(
        opts.output_rollout_dir,
        opts.run_name
    ) if not opts.no_saving else None

    opts.log_dir = os.path.join(
        opts.log_dir,
        opts.run_name
    )

    opts.test_dir = os.path.join(
        opts.test_dir,
        opts.run_name
    )

    return opts
