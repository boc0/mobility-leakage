import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
import json
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from tqdm import tqdm
from scipy.sparse import coo_matrix

# parse settings
setting = Setting()
setting.parse()
dir_name = os.path.dirname(setting.log_file)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
setting.log_file = setting.log_file + '_' + timestring
log = open(setting.log_file, 'w')

# ensure save directory exists for model checkpoints
os.makedirs(setting.save_dir, exist_ok=True)

# print(setting)

# log_string(log, 'log_file: ' + setting.log_file)
# log_string(log, 'user_file: ' + setting.trans_user_file)
# log_string(log, 'loc_temporal_file: ' + setting.trans_loc_file)
# log_string(log, 'loc_spatial_file: ' + setting.trans_loc_spatial_file)
# log_string(log, 'interact_file: ' + setting.trans_interact_file)

# log_string(log, str(setting.lambda_user))
# log_string(log, str(setting.lambda_loc))

# log_string(log, 'W in AXW: ' + str(setting.use_weight))
# log_string(log, 'GCN in user: ' + str(setting.use_graph_user))
# log_string(log, 'spatial graph: ' + str(setting.use_spatial_graph))

message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# load dataset
poi_loader = PoiDataloader(
    setting.max_users, setting.min_checkins)  # 0， 5*20+1
poi_loader.read(setting.dataset_file)
# print('Active POI number: ', poi_loader.locations())  # 18737 106994
# print('Active User number: ', poi_loader.user_count())  # 32510 7768
# print('Total Checkins number: ', poi_loader.checkins_count())  # 1278274

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN)  # 20, 200 or 1024, 0
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(
), 'batch size must be lower than the amount of available users'

# create flashback trainer
with open(setting.trans_loc_file, 'rb') as f:  # transition POI graph
    transition_graph = pickle.load(f)  # 在cpu上
# transition_graph = top_transition_graph(transition_graph)
transition_graph = coo_matrix(transition_graph)

if setting.use_spatial_graph:
    with open(setting.trans_loc_spatial_file, 'rb') as f:  # spatial POI graph
        spatial_graph = pickle.load(f)  # 在cpu上
    # spatial_graph = top_transition_graph(spatial_graph)
    spatial_graph = coo_matrix(spatial_graph)
else:
    spatial_graph = None

if setting.use_graph_user:
    with open(setting.trans_user_file, 'rb') as f:
        friend_graph = pickle.load(f)  # 在cpu上
    # friend_graph = top_transition_graph(friend_graph)
    friend_graph = coo_matrix(friend_graph)
else:
    friend_graph = None

with open(setting.trans_interact_file, 'rb') as f:  # User-POI interaction graph
    interact_graph = pickle.load(f)  # 在cpu上
interact_graph = csr_matrix(interact_graph)

log_string(log, 'Successfully load graph')

trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
                           setting.use_weight, transition_graph, spatial_graph, friend_graph, setting.use_graph_user,
                           setting.use_spatial_graph, interact_graph)  # 0.01, 100 or 1000
h0_strategy = create_h0_strategy(
    setting.hidden_dim, setting.is_lstm)  # 10 True or False

# Use graph shapes to determine model dimensions to support subset training with union-shaped graphs
model_loc_count = transition_graph.shape[0]
model_user_count = interact_graph.shape[0] if interact_graph is not None else poi_loader.user_count()
trainer.prepare(model_loc_count, model_user_count, setting.hidden_dim, setting.rnn_factory,
                setting.device)
evaluation_test = Evaluation(dataset_test, dataloader_test,
                             poi_loader.user_count(), h0_strategy, trainer, setting, log)
print('{} {}'.format(trainer, setting.rnn_factory))

#  training loop
optimizer = torch.optim.Adam(trainer.parameters(
), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')

bar = tqdm(total=setting.epochs)
bar.set_description('Training')

train_losses = []
valid_losses = []
accuracies = []  # we will use recall@1 as accuracy proxy

for e in range(setting.epochs):  # 100
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()  # shuffle users before each epoch!

    losses = []
    epoch_start = time.time()
    for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(dataloader):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)

        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)

        optimizer.zero_grad()
        loss = trainer.loss(x, t, t_slot, s, y, y_t,
                            y_t_slot, y_s, h, active_users)

        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(trainer.parameters(), 5)
        losses.append(loss.item())
        optimizer.step()

    # schedule learning rate:
    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(
        epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')

    if (e + 1) % setting.validate_epoch == 0:
        log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
        evl_start = time.time()
        metrics_eval = evaluation_test.evaluate()  # dict with recalls
        evl_end = time.time()
        log_string(log, 'One evaluate need {:.2f}s'.format(
            evl_end - evl_start))
        # compute validation loss (optional) by iterating test dataloader once
        dataset_test.reset()
        h_val = h0_strategy.on_init(setting.batch_size, setting.device)
        val_losses_ep = []
        for i_val, (xv, tv, tslotv, sv, yv, y_tv, y_tslotv, y_sv, reset_hv, active_users_v) in enumerate(dataloader_test):
            for jv, resetv in enumerate(reset_hv):
                if resetv:
                    if setting.is_lstm:
                        hc = h0_strategy.on_reset(active_users_v[0][jv])
                        h_val[0][0, jv] = hc[0]
                        h_val[1][0, jv] = hc[1]
                    else:
                        h_val[0, jv] = h0_strategy.on_reset(active_users_v[0][jv])
            xv = xv.squeeze().to(setting.device)
            tv = tv.squeeze().to(setting.device)
            tslotv = tslotv.squeeze().to(setting.device)
            sv = sv.squeeze().to(setting.device)
            yv = yv.squeeze().to(setting.device)
            y_tv = y_tv.squeeze().to(setting.device)
            y_tslotv = y_tslotv.squeeze().to(setting.device)
            y_sv = y_sv.squeeze().to(setting.device)
            active_users_v = active_users_v.to(setting.device)
            with torch.no_grad():
                lval = trainer.loss(xv, tv, tslotv, sv, yv, y_tv, y_tslotv, y_sv, h_val, active_users_v)
            val_losses_ep.append(lval.item())
        valid_loss_epoch = float(np.mean(val_losses_ep)) if val_losses_ep else 0.0
        valid_losses.append(valid_loss_epoch)
        accuracies.append(metrics_eval['recall@1'])

    # collect train loss per epoch (after epoch loop)
    train_losses.append(np.mean(losses))

    # save checkpoint at the end of each epoch
    checkpoint_epoch_path = os.path.join(
        setting.save_dir, f'flashback_{timestring}_epoch{e + 1}.pt')
    checkpoint_latest_path = os.path.join(
        setting.save_dir, f'flashback_{timestring}_latest.pt')
    torch.save({
        'epoch': e + 1,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hidden_dim': setting.hidden_dim,
        'lambda_t': setting.lambda_t,
        'lambda_s': setting.lambda_s,
        'lambda_loc': setting.lambda_loc,
        'lambda_user': setting.lambda_user,
        'loc_count': model_loc_count,
        'user_count': model_user_count,
        'rnn_type': str(setting.rnn_factory)
    }, checkpoint_epoch_path)
    # also update a rolling latest checkpoint
    torch.save({
        'epoch': e + 1,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hidden_dim': setting.hidden_dim,
        'lambda_t': setting.lambda_t,
        'lambda_s': setting.lambda_s,
        'lambda_loc': setting.lambda_loc,
        'lambda_user': setting.lambda_user,
        'loc_count': model_loc_count,
        'user_count': model_user_count,
        'rnn_type': str(setting.rnn_factory)
    }, checkpoint_latest_path)

bar.close()

# write learning curve to res.txt in save_dir
try:
    res_path = os.path.join(setting.save_dir, 'res.txt')
    metrics_view = {
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'accuracy': accuracies
    }
    args_view = {
        'hidden_dim': setting.hidden_dim,
        'lr': setting.learning_rate,
        'epochs': setting.epochs,
        'batch_size': setting.batch_size,
        'lambda_t': setting.lambda_t,
        'lambda_s': setting.lambda_s,
        'lambda_loc': setting.lambda_loc,
        'lambda_user': setting.lambda_user,
        'rnn': str(setting.rnn_factory)
    }
    with open(res_path, 'w') as f_res:
        json.dump({'args': args_view, 'metrics': metrics_view}, f_res, indent=4)
    log_string(log, f'Saved learning curve to {res_path}')
except Exception as e:
    log_string(log, f'Failed to write res.txt: {e}')
