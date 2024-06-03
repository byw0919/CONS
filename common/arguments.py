import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--env_name', type=str, default='ma_gym:PGM-3ag-v0', help='env name')
    parser.add_argument('--max_episodes', type=int, default=100000, help='total episodes')
    parser.add_argument('--warm_up_steps', type=int, default=5000, help='warm up steps')
    parser.add_argument('--save_cycle', type=int, default=10000, help='model save cycle')
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')  # load_model时要和模型匹配
    parser.add_argument('--individual_rewards', type=bool, default=True, help='whether to use individual rewards for training')
    parser.add_argument('--eps', type=float, default=1e-8, help='RMSprop eps')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    # GA-COMM: --alg central_v+g2anet
    parser.add_argument('--alg', type=str, default='cons', help='the algorithm to train the agent')
    parser.add_argument('--c_ep', type=int, default=10000, help='parameter-1 used to adjust the downward trend of negative knowledge weight')
    parser.add_argument('--c_w', type=float, default=0.5, help='parameter-2 used to adjust the downward trend of negative knowledge weight')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')  # 每次往经验池放1个episode的经验，之后训练
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='how often to evaluate the model')  # 默认5000
    parser.add_argument('--evaluate_episode', type=int, default=10, help='number of the episode to evaluate the agent')
    parser.add_argument('--test_episode', type=int, default=10, help='number of the episode to test the trained model')
    parser.add_argument('--load_model', default=False, action='store_true', help='whether to load the pretrained model')  # 使用模型时，设为True
    parser.add_argument('--evaluate', default=False, action='store_true', help='whether to evaluate the model')  # 使用模型时，设为True
    parser.add_argument('--render', default=False, action='store_true', help='whether to render the environment')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--pkl_dir', default='model/PGM-3ag/cons/', help='trained model directory')
    args = parser.parse_args()

    return args


# arguments of cons
def get_cons_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(1e4)
    args.budget = 50000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # the episode when knowledge sharing is initiated
    args.start_advice = 5000

    # update rate for action probability distribution
    args.tau = 0.6

    args.variable_a = 0.5  # scaling variable va
    return args


# arguments of adhoc_td
def get_adhoctd_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(1e4)
    args.budget = 50000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # the episode when knowledge sharing is initiated
    args.start_advice = 5000
    args.variable_a = 0.5  # scaling variable va
    args.variable_g = 1.5  # scaling variable vg
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vdn/qmix/qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(1e4)

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of reinforce
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    args.k = 1
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args

