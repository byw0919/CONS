from runner import Runner
from clean_up import ssd, config_ssd
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, get_cons_args, get_commnet_args, get_g2anet_args, get_adhoctd_args
import datetime
import torch
import numpy as np
import os
import random

torch.autograd.set_detect_anomaly = True
np.seterr(invalid='ignore')

if __name__ == '__main__':
    num_seeds = 1
    for i in range(num_seeds):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        elif args.alg.find('cons') > -1:
            args = get_cons_args(args)
        elif args.alg.find('adhoc_td') > -1:
            args = get_adhoctd_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)

        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        args.seed += i  # different seed
        config = config_ssd.get_config()
        env = ssd.Env(config.env)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        env.seed(args.seed)

        env_name = 'clean_up-10x10-4ag'
        args.curtime = current_time
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        if not (args.evaluate and args.load_model):
            result_path = './results/' + f'{current_time}_{env_name}_{args.alg}_{args.seed}/'
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            csv_path = result_path + 'csvdata/'
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)

            model_path = result_path + 'model/'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            args.model_dir = model_path

            plt_path = result_path + 'plt/'
            if not os.path.exists(plt_path):
                os.makedirs(plt_path)
            args.plt_dir = plt_path

            args.train_csv_file = csv_path + f'{current_time}_{args.alg}({str(args.reuse_network)})_{args.seed}_train.csv'
            args.eval_csv_file = csv_path + f'{current_time}_{args.alg}({str(args.reuse_network)})_{args.seed}_eval.csv'

        print('seed={}, reuse_network={}, individual_reward = {}'.
              format(args.seed, args.reuse_network, args.individual_rewards))
        if args.alg.find('cons') > -1:
            print("CONS: c_ep={}, c_w={}".format(args.c_ep, args.c_w))

        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            episode_reward = runner.evaluate()
            print('The average episode reward of {} is {}'.format(args.alg, episode_reward))

        env.close()
