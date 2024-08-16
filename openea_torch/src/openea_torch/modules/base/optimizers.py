import torch
import torch.optim as optim


def generate_optimizer(loss, learning_rate, var_list=None, opt='SGD'):
    optimizer = get_optimizer(opt, learning_rate, var_list)
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 应用梯度更新参数
    return optimizer


def get_optimizer(opt, learning_rate, var_list):
    if var_list is None:
        raise ValueError("var_list should not be None in PyTorch. Pass model parameters explicitly.")

    if opt == 'Adagrad':
        optimizer = optim.Adagrad(var_list, lr=learning_rate)
    elif opt == 'Adadelta':
        optimizer = optim.Adadelta(var_list, lr=learning_rate)
    elif opt == 'Adam':
        optimizer = optim.Adam(var_list, lr=learning_rate)
    else:  # opt == 'SGD'
        optimizer = optim.SGD(var_list, lr=learning_rate)
    return optimizer

# # 示例用法：
# class Model(nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.ent_embeds = nn.Parameter(torch.randn(args.num_entities, args.dim))
#         add_mapping_variables(self)
#         add_mapping_module(self)

#     def forward(self, seed_entities1, seed_entities2):
#         tes1 = self.ent_embeds[seed_entities1]
#         tes2 = self.ent_embeds[seed_entities2]
#         loss = self.args.alpha * mapping_loss(tes1, tes2, self.mapping_mat, self.eye_mat)
#         return loss

# args = Args()
# model = Model(args)
# optimizer = generate_optimizer(model.loss, args.learning_rate, model.parameters(), opt=args.optimizer)