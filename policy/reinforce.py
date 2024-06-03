import torch
import os
from network.base_net import RNN
from network.commnet import CommNet
from network.g2anet import G2ANet


class Reinforce:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape  # actor网络输入的维度，和vdn、qmix的rnn输入维度一样，使用同一个网络结构
        # 根据参数决定RNN的输入维度
        if args.last_action:
            actor_input_shape += self.n_actions
        self.args = args
        if not (args.evaluate and args.load_model):  # save dir
            self.model_dir = args.model_dir
        else:  # load dir
            self.model_dir = args.pkl_dir
        if args.reuse_network:
            actor_input_shape += self.n_agents
            # 神经网络
            # 每个agent选动作的网络,输出当前agent所有动作对应的概率，用该概率选动作的时候还需要用softmax再运算一次。
            if self.args.alg == 'reinforce':
                print('Init alg REINFORCE')
                self.eval_rnn = RNN(actor_input_shape, args)
            elif self.args.alg == 'reinforce+commnet':
                print('Init alg reinforce+commnet')
                self.eval_rnn = CommNet(actor_input_shape, args)
            elif self.args.alg == 'reinforce+g2anet':
                print('Init alg reinforce+g2anet')
                self.eval_rnn = G2ANet(actor_input_shape, args)
            else:
                raise Exception("No such algorithm")

            if self.args.cuda:
                self.eval_rnn.cuda()

            # 如果存在模型则加载模型
            if self.args.load_model:
                if os.path.exists(self.model_dir + 'rnn_params.pkl'):
                    path_rnn = self.model_dir + 'rnn_params.pkl'
                    map_location = 'cuda:0' if self.args.cuda else 'cpu'
                    self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                    print('Successfully load the model: {}'.format(path_rnn))
                else:
                    raise Exception("No model!")

            self.rnn_parameters = list(self.eval_rnn.parameters())
            if args.optimizer == "RMS":
                self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)
            self.args = args

            # 执行过程中，要为每个agent都维护一个eval_hidden
            # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden
            self.eval_hidden = None
        else:
            self.eval_rnn, self.eval_hidden = [], []
            self.rnn_parameters, self.rnn_optimizer = [], []
            for i in range(self.n_agents):
                if self.args.alg == 'reinforce':
                    self.eval_rnn.append(RNN(actor_input_shape, args))
                self.eval_hidden.append(None)
                if self.args.cuda:
                    self.eval_rnn[i].cuda()

                if self.args.load_model:
                    if os.path.exists(self.model_dir + 'rnn_net_params_' + str(i) + '.pkl'):
                        path_rnn = self.model_dir + 'rnn_net_params_' + str(i) + '.pkl'
                        map_location = 'cuda:0' if self.args.cuda else 'cpu'
                        self.eval_rnn[i].load_state_dict(torch.load(path_rnn, map_location=map_location))
                        print('Successfully load the model: {}'.format(path_rnn))
                    else:
                        raise Exception("No model!")

                self.rnn_parameters.append(list(self.eval_rnn[i].parameters()))
                if args.optimizer == "RMS":
                    self.rnn_optimizer.append(torch.optim.RMSprop(self.rnn_parameters[i], lr=args.lr_actor))
            print('Init alg REINFORCE')

    def learn(self, batch, max_episode_len, train_step, epsilon):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated = batch['u'], batch['r'],  batch['avail_u'], batch['terminated']
        mask = (1 - batch["padded"].float())  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            r = r.cuda()
            u = u.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()

        # 得到每条经验的return, (episode_num, max_episode_len， n_agents)
        n_return = self._get_returns(r, mask, terminated, max_episode_len)

        # 每个agent的所有动作的概率 (episode_num, max_episode_len， n_agents，n_actions)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        if self.args.reuse_network:
            # 给mask转换出n_agents维度，用于每个agent的训练
            mask = mask.repeat(1, 1, self.n_agents)
            # 每个agent的选择的动作对应的概率 (episode_num, max_episode_len， n_agents)
            pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # (1, max_episode_len, n_ag, n_act) --> (1, max_episode_len, n_ag)
            pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
            log_pi_taken = torch.log(pi_taken)

            # loss函数，(episode_num, max_episode_len, n_agents)
            # F.cross_entropy(input = act_prob,target=torch.LongTensor(self.ep_act).to(device),reduction = 'none')
            loss = - ((n_return * log_pi_taken) * mask).sum() / mask.sum()
            self.rnn_optimizer.zero_grad()
            loss.backward()
            if self.args.alg == 'reinforce+g2anet':
                torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
            self.rnn_optimizer.step()
            return loss
            # print('Actor loss is', loss)
        else:
            u = u.permute(2, 0, 1, 3)
            mask = mask.repeat(1, 1, self.n_agents).permute(2, 0, 1)
            all_loss = []
            for i in range(self.n_agents):
                pi_taken = torch.gather(action_prob[i], dim=2, index=u[i]).squeeze(2)
                pi_taken[mask[i] == 0] = 1.0
                log_pi_taken = torch.log(pi_taken)

                loss = - ((n_return[i] * log_pi_taken) * mask).sum() / mask.sum()
                all_loss.append(loss)
                self.rnn_optimizer[i].zero_grad()
                loss.backward()
                if self.args.alg == 'reinforce+g2anet':
                    torch.nn.utils.clip_grad_norm_(self.rnn_parameters[i], self.args.grad_norm_clip)
                self.rnn_optimizer[i].step()
            return sum(all_loss)

    def _get_returns(self, r, mask, terminated, max_episode_len):
        r = r.squeeze(-1)  # 传进来的时候是(1, max_episode_len, n_agent),处理之后变还是这样
        mask = mask.squeeze(-1)  # (1, max_episode, 1) --> (1, max_episode_len)
        terminated = terminated.squeeze(-1)  # (1, max_episode, 1) --> (1, max_episode_len)
        terminated = 1 - terminated  # (0, 0, 0, ... , 1) --> (1, 1, ... , 0)
        n_return = torch.zeros_like(r)  # (1, max_episode_len, n_agent)
        n_return[:, -1] = r[:, -1] * mask[:, -1]  # 首先是最后一步
        for transition_idx in range(max_episode_len - 2, -1, -1):
            n_return[:, transition_idx] = (r[:, transition_idx] + self.args.gamma * n_return[:, transition_idx + 1] * terminated[:, transition_idx]) * mask[:, transition_idx]
        if self.args.reuse_network:
            return n_return.unsqueeze(-1).expand(-1, -1, self.n_agents)
        else:
            return n_return.permute(2, 0, 1)

    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        # 给inputs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
            # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
            inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        else:
            # (episode_num, n_agents, n_obs).permute(1, 0, 2)=(n_agents, episode_num, n_obs)
            inputs = torch.cat(inputs, dim=2).permute(1, 0, 2)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        if self.args.reuse_network:
            for transition_idx in range(max_episode_len):
                inputs = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
                if self.args.cuda:
                    inputs = inputs.cuda()
                    self.eval_hidden = self.eval_hidden.cuda()
                # inputs维度为(episode_num * n_agents,inputs_shape)，得到的outputs维度为(episode_num * n_agents, n_actions)
                outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
                # 把q_eval维度重新变回(8, 5,n_actions)
                outputs = outputs.view(episode_num, self.n_agents, -1)
                prob = torch.nn.functional.softmax(outputs, dim=-1)
                action_prob.append(prob)
            # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
            # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
            # action_prob里面是max_episode_len个(1, n_agent, n_act)
            action_prob = torch.stack(action_prob, dim=1).cpu()

            action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])   # 可以选择的动作的个数
            action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
            action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

            # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
            action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
            # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
            # 因此需要再一次将该经验对应的概率置为0
            action_prob[avail_actions == 0] = 0.0
            if self.args.cuda:
                action_prob = action_prob.cuda()
            return action_prob
        else:
            for i in range(self.n_agents):
                prob_i = []
                for transition_idx in range(max_episode_len):
                    inputs = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
                    if self.args.cuda:
                        inputs = inputs.cuda()
                        self.eval_hidden[i] = self.eval_hidden[i].cuda()
                    q, self.eval_hidden[i] = self.eval_rnn[i](inputs[i], self.eval_hidden[i])  # (1, n_act)
                    prob = torch.nn.functional.softmax(q, dim=-1)  # (1, n_act)
                    prob_i.append(prob)  # max_episode_len个(1, n_act)
                prob_i = torch.stack(prob_i, dim=1).cpu()  # (1, max_episode_len, n_act)

                avail_actions_i = avail_actions.permute(2,0,1,3)[i]
                action_num = avail_actions_i.sum(dim=-1, keepdim=True).float().repeat(1, 1, avail_actions_i.shape[-1])  # 可以选择的动作的个数
                prob_i = ((1 - epsilon) * prob_i + torch.ones_like(prob_i) * epsilon / action_num)
                prob_i[avail_actions_i == 0] = 0.0  # 不能执行的动作概率为0
                prob_i = prob_i / prob_i.sum(dim=-1, keepdim=True)
                prob_i[avail_actions_i == 0] = 0.0
                if self.args.cuda:
                    prob_i = prob_i.cuda()
                action_prob.append(prob_i)  # n_agents个(1, max_episode_len, n_act)
            return action_prob

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden
        if self.args.reuse_network:
            self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        else:
            for i in range(self.n_agents):
                self.eval_hidden[i] = torch.zeros((episode_num, self.args.rnn_hidden_dim))


    def save_model(self, train_step, episode_counts):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.args.reuse_network:
            torch.save(self.eval_rnn.state_dict(), f"{self.model_dir}/{num}({episode_counts})_rnn_net_params.pkl")
        else:
            for i in range(self.n_agents):
                torch.save(self.eval_rnn[i].state_dict(),f"{self.model_dir}/{num}({episode_counts})_rnn_net_params_{str(i)}.pkl")