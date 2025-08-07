import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributions as td

# class MultiTemporalInterventionMatrix(pl.LightningModule):
#     def __init__(
#             self,
#             num_nodes: int,
#             num_inter: int,
#             min_num_tar: int = 1,
#             max_num_tar: int = 3,
#             tau_gumbel: float = 1.0,
#             reg_weight: float = 1.0  # 正则化强度
#     ):
#         super().__init__()
#         self.num_nodes = num_nodes
#         #self.lag = lag
#         self.num_inter = num_inter
#         self.min_num_tar = min_num_tar
#         self.max_num_tar = max_num_tar
#         self.tau_gumbel = tau_gumbel
#         self.reg_weight = reg_weight
#
#         # 每个位置独立的logits (K, L+1, N)
#         self.logits = nn.Parameter(torch.zeros(num_inter, num_nodes),requires_grad=True)
#
#     def sample_intermatrix(self, hard=True):
#         """ 使用Gumbel-Sigmoid采样干预矩阵 """
#         # 生成Gumbel噪声
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
#
#         # 添加噪声并计算概率
#         perturbed_logits = (self.logits + gumbel_noise) / self.tau_gumbel
#         probs = torch.sigmoid(perturbed_logits)
#
#         if hard:
#             # 直通估计器：前向传播使用硬阈值，反向传播使用概率梯度
#             inter_matrix = (probs >= 0.5).float()
#             inter_matrix = inter_matrix + (probs - probs.detach())  # 保持梯度
#         else:
#             inter_matrix = probs  # 软采样
#
#         return inter_matrix
#
#     def regularization_loss(self):
#         """ 计算期望目标数量的正则化损失 """
#         probs = torch.sigmoid(self.logits)  # (K, L+1, N)
#         expected_sum = probs.sum(dim=-1)  # (K, L+1)
#         # 计算超出范围的惩罚
#         lower_penalty = F.relu(self.min_num_tar - expected_sum)
#         upper_penalty = F.relu(expected_sum - self.max_num_tar)
#         # 总正则损失
#         reg_loss = (lower_penalty + upper_penalty).mean()
#         return self.reg_weight * reg_loss
#
#     def forward(self):
#         return self.sample_intermatrix()
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters())
import torch.distributions as dist


class MultiTemporalInterventionMatrix(pl.LightningModule):
    def __init__(self, M: int, d: int = 5, alpha: float = 1.0,tau_gumbel: float = 1.0):
        super().__init__()
        self.M = M  # 干预上下文数
        self.d = d  # 变量数
        self.alpha = alpha  # 温度参数
        self.tau_gumbel = tau_gumbel
        # 潜在变量Γ，初始化为较小值以接近均匀分布
        # self.gamma_prior_std = (0.1) ** 0.5
        # self.gamma = nn.Parameter(torch.normal(mean=0.0, std=self.gamma_prior_std, size=(self.M, self.d)),requires_grad=True)
        #self.gamma = nn.Parameter(torch.randn(M, d), requires_grad=True)
        #self.logits = nn.Parameter(torch.zeros((2, M,d),device=self.device),requires_grad=True)
        self.logits = nn.Parameter((torch.ones(self.M, self.d+1,
                           device=self.device) * 0.01
        ), requires_grad=True)
        # Beta分布参数（固定或可学习）
        self.zeta1 = 1.0 / self.d
        # self.zeta2 = (d - 1.0) / self.d
        self.zeta2=1-self.zeta1
    def sample_intermatrix(self,idx):
        # 计算干预概率（带温度调节的sigmoid）
        #I_probs = torch.sigmoid(self.gamma / self.alpha)
        #I_probs = torch.sigmoid(self.alpha/2*self.gamma)
        # I_probs_combined = torch.stack([ 1 - I_probs,I_probs], dim=0)
        # inter_matrix=F.gumbel_softmax(self.logits,
        #                  tau=self.tau_gumbel,
        #                  hard=True,  # 输出是onehot向量
        #                  dim=0)[1, ...]
        inter_matrix = F.gumbel_softmax(self.logits,
                                        tau=self.tau_gumbel,
                                        hard=True,  # 输出是onehot向量
                                        dim=1)
        is_zero=inter_matrix[:,0].unsqueeze(-1)
        positions=inter_matrix[:,1:]
        final_matrix=positions*(1-is_zero)
        #inter_matrix = torch.bernoulli(I_probs.new_ones((self.M, self.d)) * I_probs).int()
        #print("inter_matrix:\n",inter_matrix)
        # # 采样干预目标（Straight-Through Estimator）
        # I_hard = (I_probs > 0.5).float()
        # I_soft = I_probs + (I_hard - I_probs).detach()  # 保持梯度
        #inter_matrix = inter_matrix + (probs - probs.detach())
        return final_matrix[idx]

    def get_adj(self):
        probs=F.softmax(self.logits,dim=1)
        is_zero_prob = probs[:, 0].unsqueeze(-1)  # 全0行的概率
        positions_probs = probs[:, 1:]  # 各位置的概率
        adj_matrix = positions_probs * (1 - is_zero_prob)
        #probs=torch.sigmoid(self.gamma / self.alpha)
        #probs = torch.sigmoid(self.alpha / 2 * self.gamma)
        return adj_matrix
    def prior_loss(self,inter_matrix):
        # Beta分布KL散度项（近似）
        #probs = torch.sigmoid(self.gamma)
        #probs = torch.sigmoid(self.gamma / self.alpha)
        # probs=self.get_adj()
        # probs=probs.clamp(min=1e-10,max=1-1e-10)
        # log_likelihood=(self.zeta1-1)*torch.log(probs)+(self.zeta2-1)*torch.log(1-probs)
        # log_likelihood=log_likelihood.sum()
        probs= F.softmax(self.logits, dim=1)
        #print("probs:\n",probs)
        beta_prior = dist.Beta(self.zeta1, self.zeta2)#创建一个带两个参数的Beta分布
        kl_beta = -beta_prior.log_prob(probs).sum()#Beta分布的对数概率的和
        #print("beta_prior.log_prob(probs)\n",beta_prior.log_prob(probs))
        # 高斯先验项
        #kl_gaussian = -dist.Normal(0, 1.0).log_prob(self.gamma).mean()
        # L1稀疏正则项
        return kl_beta
        # return -log_likelihood

    def entropy(self):
        # dist = td.Independent(td.Bernoulli(
        #     logits=self.logits[1, :] - self.logits[0, :]), 1)
        probs = self.get_adj()
        dist = td.Independent(td.Bernoulli(
            logits=probs-(1-probs)), 1)
        entropies= dist.entropy().sum()
        return entropies

    def forward(self):
        return self.sample_intermatrix()
