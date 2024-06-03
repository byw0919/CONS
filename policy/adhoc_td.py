import torch
import os
from network.base_net import RNN
import torch.nn.functional as F
import math
import random
from collections import Counter
import numpy as np


class AdhocTD:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        self.args = args
        if not (args.evaluate and args.load_model):  # save dir
            self.model_dir = args.model_dir
        else:  # load dir
            self.model_dir = args.pkl_dir

        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

            # 神经网络
            self.eval_rnn = RNN(input_shape, args)
            self.target_rnn = RNN(input_shape, args)
            if self.args.cuda:
                self.eval_rnn.cuda()
                self.target_rnn.cuda()
            # 如果存在模型则加载模型
            if self.args.load_model:
                if os.path.exists(self.model_dir + 'rnn_net_params.pkl'):
                    path_rnn = self.model_dir + 'rnn_net_params.pkl'
                    map_location = 'cuda:0' if self.args.cuda else 'cpu'
                    self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                    print('Successfully load the model: {}'.format(path_rnn))
                else:
                    raise Exception("No model!")

            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())

            self.eval_parameters = list(self.eval_rnn.parameters())
            if args.optimizer == "RMS":
                self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr, eps=args.eps)

            self.eval_hidden = None
            self.target_hidden = None
        else:
            self.eval_rnn, self.target_rnn, self.eval_hidden, self.target_hidden = [], [], [], []
            self.eval_parameters, self.optimizer = [], []
            for i in range(self.n_agents):
                self.eval_rnn.append(RNN(input_shape, args))
                self.target_rnn.append(RNN(input_shape, args))
                self.eval_hidden.append(None)
                self.target_hidden.append(None)
                if self.args.cuda:
                    self.eval_rnn[i].cuda()
                    self.target_rnn[i].cuda()
                if self.args.load_model:
                    if os.path.exists(self.model_dir + 'rnn_net_params_' + str(i) + '.pkl'):
                        path_rnn = self.model_dir + 'rnn_net_params_' + str(i) + '.pkl'
                        map_location = 'cuda:0' if self.args.cuda else 'cpu'
                        self.eval_rnn[i].load_state_dict(torch.load(path_rnn, map_location=map_location))
                        print('Successfully load the model: {}'.format(path_rnn))
                    else:
                        raise Exception("No model!")

                self.target_rnn[i].load_state_dict(self.eval_rnn[i].state_dict())
                self.eval_parameters.append(list(self.eval_rnn[i].parameters()))
                if args.optimizer == "RMS":
                    self.optimizer.append(torch.optim.RMSprop(self.eval_parameters[i], lr=args.lr, eps=args.eps))
        random.seed(args.seed)
        print('Init alg Adhoc_TD')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, avail_u, avail_u_next, terminated = batch['u'], batch['avail_u'], batch['avail_u_next'], \
                                               batch['terminated'].repeat(1, 1, self.n_agents)
        if self.args.individual_rewards:
            r = batch['r']
        else:
            r = batch['r'].repeat(1, 1, self.n_agents)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            u = u.cuda()  # (episode_num, max_episode_len, n_agents, 1)
            r = r.cuda()  # (episode_num, max_episode_len, n_agents)
            terminated = terminated.cuda()  # (episode_num, max_episode_len, n_agents)
            mask = mask.cuda()  # (episode_num, max_episode_len, n_agents)
        if self.args.reuse_network:
            # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
            q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
            # 得到target_q
            q_targets[avail_u_next == 0.0] = - 9999999
            q_targets = q_targets.max(dim=3)[0]  # (episode_num, episode_len, self.n_agents)

            targets = r + self.args.gamma * q_targets * (1 - terminated)
            td_error = (q_evals - targets.detach())
            masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

            # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
            loss = (masked_td_error ** 2).sum() / mask.sum()
            # print('loss is ', loss)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
            self.optimizer.step()

            if train_step > 0 and train_step % self.args.target_update_cycle == 0:
                self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            return loss
        else:
            u = u.permute(2, 0, 1, 3)
            avail_u_next = avail_u_next.permute(2, 0, 1, 3)  # avail_u_next原来为(1,100,5,5)
            r = r.permute(2, 0, 1)
            terminated = terminated.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            all_loss = []
            for i in range(self.n_agents):
                # 取智能体i动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
                # q_evals: reuse:(episode_num,100, n_agents, n_actions)
                # q_evals: not reuse:(episode_num, 100, n_actions)
                q_evals[i] = torch.gather(q_evals[i], dim=2, index=u[i]).squeeze(2)  # u[i]是(episode_num,100,1),q_evals[i]是(episode_num,100,n_actions),squeeze之前是(episode_num,100,1),之后是(episode_num,100)

                # 得到target_q
                # 得到target_q max_acts: episode_num, episode_limit, n_agent, 1 (与qtarget前3维shape相同)
                q_targets[i][avail_u_next[i] == 0.0] = - 9999999
                q_targets[i] = q_targets[i].max(dim=2)[0]  # (episode_num,episode_len)

                targets = r[i] + self.args.gamma * q_targets[i] * (1 - terminated[i])  # (episode_num,100)
                td_error = (q_evals[i] - targets.detach())  # (episode_num,100)
                masked_td_error = mask[i] * td_error  # 抹掉填充的经验的td_error  (episode_num,100)

                # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
                loss = (masked_td_error ** 2).sum() / mask[i].sum()  # 一个数，维度是[]
                all_loss.append(loss)

                self.optimizer[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.eval_parameters[i], self.args.grad_norm_clip)
                self.optimizer[i].step()

            if train_step > 0 and train_step % self.args.target_update_cycle == 0:
                for i in range(self.n_agents):
                    self.target_rnn[i].load_state_dict(self.eval_rnn[i].state_dict())
            return sum(all_loss)

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]  # obs:Tensor(episode_num, n_agents, n_obs)
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))  # append:(episode_num, n_agents, n_actions)
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1)) # append: (episode_num, n_agents, n_agents)
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
            inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        else:
            inputs = torch.cat(inputs, dim=2).permute(1, 0, 2)  # (episode_num, n_agents, n_obs)
            inputs_next = torch.cat(inputs_next, dim=2).permute(1, 0, 2)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        if self.args.reuse_network:
            for transition_idx in range(max_episode_len):
                inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
                if self.args.cuda:
                    inputs = inputs.cuda()
                    inputs_next = inputs_next.cuda()
                    self.eval_hidden = self.eval_hidden.cuda()
                    self.target_hidden = self.target_hidden.cuda()
                q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
                q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

                q_eval = q_eval.view(episode_num, self.n_agents, -1)
                q_target = q_target.view(episode_num, self.n_agents, -1)
                q_evals.append(q_eval)
                q_targets.append(q_target)
            # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
            # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
            q_evals = torch.stack(q_evals, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            for i in range(self.n_agents):  # 准备让q_evals中存5个智能体的100个转移的q值
                q_eval_i = []
                q_target_i = []
                for transition_idx in range(max_episode_len):
                    inputs, inputs_next = self._get_inputs(batch, transition_idx)  # (episode_num, n_agents, n_obs)
                    if self.args.cuda:
                        inputs = inputs.cuda()
                        inputs_next = inputs_next.cuda()
                        self.eval_hidden[i] = self.eval_hidden[i].cuda()
                        self.target_hidden[i] = self.target_hidden[i].cuda()
                    q, self.eval_hidden[i] = self.eval_rnn[i](inputs[i], self.eval_hidden[i])  # inputs维度为(n_agents*episode_num,n_obs)，self.eval_hidden维度为(episode_num,n_agents,64),得到的q_eval维度为(n_agents*episode_num,n_actions)
                    q_t, self.target_hidden[i] = self.target_rnn[i](inputs_next[i], self.target_hidden[i])
                    q_eval_i.append(q)
                    q_target_i.append(q_t)
                q_eval_i = torch.stack(q_eval_i, dim=1)
                q_target_i = torch.stack(q_target_i, dim=1)
                q_evals.append(q_eval_i)
                q_targets.append(q_target_i)

        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        if self.args.reuse_network:
            self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
            self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        else:
            for i in range(self.n_agents):
                self.eval_hidden[i] = torch.zeros((episode_num, self.args.rnn_hidden_dim))
                self.target_hidden[i] = torch.zeros((episode_num, self.args.rnn_hidden_dim))

    def save_model(self, train_step, episode_counts):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.args.reuse_network:
            torch.save(self.eval_rnn.state_dict(), f"{self.model_dir}/{num}({episode_counts})_rnn_net_params.pkl")
        else:
            for i in range(self.n_agents):
                torch.save(self.eval_rnn[i].state_dict(), f"{self.model_dir}/{num}({episode_counts})_rnn_net_params_{str(i)}.pkl")

    def select_from_advice(self, advised_list):
        if len(advised_list) == 0:
            return None
        elif len(advised_list) == 1:
            return advised_list[0]
        else:
            count = Counter(advised_list).most_common()
            action_index = []
            most_times = count[0][1]
            for tuple in count:
                if tuple[1] < most_times:
                    break
                if tuple[1] == most_times:
                    action_index.append(tuple[0])
            action = random.choices(action_index, k=1)[0]
        return action

    def ask_advice(self, out, hidden_state, obs, last_action, i, agent_obs, epi_obs, agent_budge, episode_counts):  # 传进的out是q_value，未做softmax
        out = F.softmax(out, dim=-1)
        agent_id = np.zeros(self.n_agents)
        agent_id[i] = 1.
        # whether to ask for advice

        if random.random() < math.pow((1 + self.args.variable_a), -math.sqrt(agent_obs[i][tuple(obs)])) and obs not in np.array(epi_obs[i]):
            agent_budge[i] = agent_budge[i] - 1
            s, s_key = [], []  # (n_agents, n_obs)
            for j in range(self.n_agents):
                s.append(obs)
                s_key.append(obs)
                if self.args.last_action:
                    s[j] = np.hstack((s[j], last_action))  #
                if self.args.reuse_network:
                    s[j] = np.hstack((s[j], np.eye(self.n_agents)[j]))
            obs_adv = torch.Tensor(np.array(s))
            if self.args.cuda:
                obs_adv = obs_adv.cuda()
                out_adv = torch.zeros([self.n_agents, self.n_actions]).cuda()
            else:
                out_adv = torch.zeros([self.n_agents, self.n_actions])
            if self.args.reuse_network:
                out_adv, _ = self.eval_rnn(obs_adv, hidden_state.repeat(self.n_agents, 1))
            else:
                for advisor_i in range(self.n_agents):
                    out_adv[advisor_i], _ = self.eval_rnn[advisor_i](obs_adv[advisor_i].unsqueeze(0), hidden_state)
            out_adv = F.softmax(out_adv, dim=-1).detach()  # out_adv: [n_agent, n_actions]

            advised_actions = []
            for j in range(self.n_agents):
                if i == j:
                    continue
                if self.args.alg.find('v1'):
                    advised_actions.append(out_adv[j, :].argmax().item())
                else:
                    difQ = math.fabs(out_adv[j, :].max() - out_adv[j, :].min())
                    numberVisits = 0 if tuple(s_key[j]) not in agent_obs[j] else agent_obs[j][tuple(s_key[j])]
                    value = (math.sqrt(numberVisits) * difQ)
                    prob = 1 - (math.pow((1 + self.args.variable_g), -value))

                    if random.random() < prob:
                        advised_actions.append(out_adv[j, :].argmax().item())

            action = self.select_from_advice(advised_actions)
            if action is None:
                agent_budge[i] = agent_budge[i] + 1
            return action, agent_budge

        action = None
        return action, agent_budge
