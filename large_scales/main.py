
from random import choice
import random
import numpy as np

import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim

import datetime
import time

# wandb.init(project='MaxEnt_ARL')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def train(args, device, X_train, Y_train):
    start_time = time.time()
    print(f'using {device}')
    seed_everything(args.seed)

    # saving file names define
    if args.save_name == 'default':
        model_file_name = args.data_name + '_' + args.model_name + '.pth'
    else:
        model_file_name = args.save_name + '.pth'
    args.down_save_name = model_file_name

    data = None    
        
    model = args.model
    model.to(device)
    

def train_farconvae(args, model, train_loader, test_loader, device, model_file_name):

    current_iter = 0

    # model / optimizer setup
    opt_clf = torch.optim.Adam(clf_xy.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    opt_vae, opt_pred = model.get_optimizer()


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_clf_acc, best_pred_acc, best_to, patience = -1e7, -1e7, -1e7, 0
    for epoch in range(1, args.epochs + 1):
        # ------------------------------------
        # Training
        # ------------------------------------
        model.train()
        clf_xy.train() if args.clf_path == 'no' else clf_xy.eval()

        # tracking quantities (loss, variance, divergence, performance, ...)
        ep_tot_loss, ep_recon_loss, ep_kl_loss, ep_c_loss, ep_sr, \
        ep_pred_loss, ep_clf_loss, ep_pred_cor, ep_clf_cor, ep_tot_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

        for x, s, y in train_loader:
            x, s, y = x.to(device), s.to(device), y.to(device)

            current_iter += 1
            n = x.shape[0]
            ori_x, ori_s, ori_y = x, s, y
            cont_x, cont_s, cont_y = x, 1-s, y
            
            # BestClf loss
            y_hat = clf_xy(ori_x)
            clf_loss = clf_lossf(y_hat, ori_y)
            y_hat_bin = torch.ones_like(y_hat, device=device)
            y_hat_bin[y_hat < 0] = 0.0

            # out : (zx, zs), (x_recon, s_recon), (mu_x, logvar_x, mu_s, logvar_s), y_pred
            out1 = model(ori_x, ori_s, ori_y)
            out2 = model(cont_x, cont_s, cont_y)

            recon_loss, kl_loss, pred_loss, cont_loss, sr_loss, y_pred = farcon_lossf(out1, out2, ori_x, cont_x, ori_s, cont_s, y, model, current_iter)

            opt_vae.zero_grad(set_to_none=True)
            opt_pred.zero_grad(set_to_none=True)
            if args.clf_path == 'no':
                opt_clf.zero_grad(set_to_none=True)

            loss = recon_loss + pred_loss + kl_loss + cont_loss + sr_loss
            loss.backward()
            if args.clf_path == 'no':
                clf_loss.backward()

            if args.clip_val != 0:
                torch.nn.utils.clip_grad_norm_(model.vae_params(), max_norm=args.clip_val)
            opt_vae.step()
            opt_pred.step()
            if args.clf_path == 'no':
                opt_clf.step()

            if (args.scheduler == 'one'):
                for sche in schedulers:
                    sche.step()

            # monitoring
            ep_tot_num += n
            ep_tot_loss += loss.item() * n
            ep_recon_loss += recon_loss.item() * n
            ep_kl_loss += kl_loss.item() * n
            ep_pred_loss += pred_loss.item() * n
            ep_pred_cor += (y_pred == ori_y).sum().item()
            ep_clf_loss += clf_loss.item() * n
            ep_clf_cor += (y_hat_bin == ori_y).sum().item()
            if args.alpha != 0:
                ep_c_loss += cont_loss.item() * n
            if args.gamma != 0:
                ep_sr += sr_loss.item() * n

        if args.scheduler == 'lr':
            for sche in schedulers:
                sche.step()
            lrs.append(opt_vae.param_groups[0]['lr'])

        # ------------------------------------
        # validation
        # ------------------------------------
        model.eval()
        ep_recon_x_loss_test, ep_recon_s_loss_test, ep_pred_loss_test, ep_clf_loss_test = 0.0, 0.0, 0.0, 0.0
        ep_tot_test_num, ep_pred_cor_test, ep_clf_cor_test, ep_c_loss_test, current_to = 0, 0, 0, 0.0, 0.0
        y_pred_raw = torch.tensor([], device=device)
        with torch.no_grad():
            for x, s, y in test_loader:
                x, s, y = x.to(device), s.to(device), y.to(device)
                n = x.shape[0]
                clf_loss_test = clf_lossf(clf_xy(x), y)
                # use binary for pseudo label in test time encoder input y
                y_hat = clf_xy(x)
                y_hat_bin = torch.ones_like(y_hat, device=device)
                y_hat_bin[y_hat < 0] = 0.0

                out1 = model(x, s, y_hat_bin)
                out2 = model(x, 1-s, y_hat_bin)

                recon_x_loss_te, recon_s_loss_te, pred_loss_te, cont_loss_te, y_pred_te = farcon_lossf(out1, out2, x, x, s, 1-s, y, model, current_iter, is_train=False)
                
                ep_tot_test_num += n
                y_pred_raw = torch.cat((y_pred_raw, y_pred_te))
                ep_recon_x_loss_test += recon_x_loss_te.item() * n
                ep_recon_s_loss_test += recon_s_loss_te.item() * n
                ep_c_loss_test += args.alpha * cont_loss_te.item() * n
                ep_pred_loss_test += pred_loss_te.item() * n
                ep_pred_cor_test += (y_pred_te == y).sum().item()
                ep_clf_cor_test += (y_hat_bin == y).sum().item()
                ep_clf_loss_test += clf_loss_test.item() * n

        # save best CLF w.r.t epoch best accuracy
        if args.clf_path == 'no':
            if best_clf_acc < ep_clf_cor_test:
                best_clf_acc = ep_clf_cor_test
                torch.save(clf_xy.state_dict(), os.path.join(args.model_path, 'clf_' + model_file_name))

        # save farcon w.r.t epoch best trade off
        ep_pred_test_acc = ep_pred_cor_test /ep_tot_test_num
        current_to = ep_pred_test_acc - (ep_c_loss_test / ep_tot_test_num)
        if best_to < current_to:
            best_to = current_to
            best_pred_acc = ep_pred_test_acc
            torch.save(model.state_dict(), os.path.join(args.model_path, 'farcon_' + model_file_name))
            patience = 0
        else:
            if args.early_stop:
                patience += 1
                if (patience % 10) == 0 :
                    print(f"----------------------------------- increase patience {patience}/{args.patience}")
                if patience > args.patience:
                    print(f"----------------------------------- early stopping")
                    break
            else:
                pass

        # -------------------------------------
        # Monitoring entire process
        # -------------------------------------
        # log_dict = collections.defaultdict(float)
        # log_dict['loss'] = ep_tot_loss / ep_tot_num
        # log_dict['recon_loss'] = (ep_recon_loss / ep_tot_num)
        # log_dict['recon_s_loss_test'] = (ep_recon_s_loss_test / ep_tot_test_num)
        # log_dict['recon_x_loss_test'] = (ep_recon_x_loss_test / ep_tot_test_num)
        # log_dict['kl_loss'] = (ep_kl_loss / ep_tot_num)
        # log_dict['c_loss'] = (ep_c_loss / ep_tot_num)
        # log_dict['pred_loss'] = (ep_pred_loss / ep_tot_num)
        # log_dict['clf_loss'] = (ep_clf_loss / ep_tot_num)
        # log_dict['pred_acc'] = (ep_pred_cor / ep_tot_num)
        # log_dict['clf_acc'] = (ep_clf_cor / ep_tot_num)
        # log_dict['ms_loss'] = (ep_sr / ep_tot_num)
        # log_dict['pred_loss_test'] = (ep_pred_loss_test / ep_tot_test_num)
        # log_dict['clf_loss_test'] = (ep_clf_loss_test / ep_tot_test_num)
        # log_dict['pred_acc_test'] = (ep_pred_cor_test / ep_tot_test_num)
        # log_dict['clf_acc_test'] = (ep_clf_cor_test / ep_tot_test_num)
        # wandb.log(log_dict)

        if (epoch % 10) == 0:
            print(f'\nEp: [{epoch}/{args.epochs}] (TRAIN) ---------------------\n'
                f'Loss: {ep_tot_loss / ep_tot_num:.3f}, L_rx: {ep_recon_loss / ep_tot_num:.3f}, '
                f'L_kl: {ep_kl_loss / ep_tot_num:.3f}, L_c: {ep_c_loss / ep_tot_num:.3f}, \n'
                f'L_clf: {ep_clf_loss / ep_tot_num:.3f}, L_pred: {ep_pred_loss / ep_tot_num:.3f}, '
                f'L_clf_acc: {ep_clf_cor / ep_tot_num:.3f}, L_pred_acc: {ep_pred_cor / ep_tot_num:.3f}')
            print(f'Ep: [{epoch}/{args.epochs}] (VALID)  ---------------------\n'
                f'L_rx: {ep_recon_x_loss_test / ep_tot_test_num:.3f}, L_rs: {ep_recon_s_loss_test / ep_tot_test_num:.3f}, '
                f'L_pred: {ep_pred_loss_test / ep_tot_test_num:.3f}, L_clf: {ep_clf_loss_test / ep_tot_test_num:.3f}, \n'
                f'L_clf_acc: {ep_clf_cor_test / ep_tot_test_num:.3f}, L_pred_acc: {ep_pred_cor_test / ep_tot_test_num:.3f}')

    # log_dict['best_pred_test_acc'] = best_pred_acc

    # return best model !
    if args.clf_path == 'no':
        clf_xy = BestClf(args.n_features, args.y_dim, args.clf_hidden_units, args)
        clf_xy.load_state_dict(torch.load(os.path.join(args.model_path, 'clf_' + model_file_name), map_location=device))
        clf_xy.to(device)

    if args.last_epmod :
        return model, clf_xy
    else:
        model = FarconVAE(args, device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'farcon_' + model_file_name), map_location=device))
        model.to(device)
        return model, clf_xy
    



def evaluation(args, device, eval_model, model=None, clf_xy=None, is_e2e=True, trainset=None, testset=None):
    seed_everything(args.seed)
    print(f'using {device}')

    train_loader, test_loader = get_xsy_loaders(os.path.join(args.data_path, args.train_file_name),
                                                    os.path.join(args.data_path, args.test_file_name),
                                                    args.data_name, args.sensitive, args.batch_size_te, args)
    testset = test_loader.dataset
    
    
    # Validate
    if is_e2e:
        print('do End-to-End Experiment phase')
        y_pred, y_acc, s_acc, s_pred = z_evaluator(args, model, clf_xy, device, eval_model, args.model_name, trainset, testset)
    else:
        model = FarconVAE(args, device)
        clf_xy = BestClf(args.n_features, 1, args.hidden_units, args)
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_file), map_location=device))
        # clf_xy.load_state_dict(torch.load(os.path.join(args.model_path, 'clf_'+args.model_file), map_location=device))
        clf_xy.load_state_dict(torch.load('./bestclf/bestclf_german.pth', map_location=device))
        
        model, clf_xy = model.to(device), clf_xy.to(device)

        # predict Y, S from learned representation using eval_model(LR or MLP) (s is always evaluated with MLP according to previous works)
        y_pred, y_acc, s_acc, s_pred = z_evaluator(args, model, clf_xy, device, eval_model, args.model_name, trainset, testset)

    # Final reporting
    if args.data_name != 'yaleb':
        dp, _, _, eodd, gap = metric_scores(os.path.join(args.data_path, args.test_file_name), y_pred)
        print('----------------------------------------------------------------')
        print(f'DP: {dp:.4f} | EO: {eodd:.4f} | GAP: {gap:.4f} | yAcc: {y_acc:.4f} | sAcc: {s_acc:.4f}')
        print('----------------------------------------------------------------')
        #performance_log = {f'DP': dp, f'EO':eodd, f'GAP': gap, f'y_acc': y_acc, f's_acc': s_acc}
        #wandb.log(performance_log)
    else:
        keys = np.unique(np.array(s_pred), return_counts=True)[0]
        values = np.unique(np.array(s_pred), return_counts=True)[1]/s_pred.shape[0]
        s_pred_log = {'pred_'+str(i):0 for i in range(5)}
        for i in range(len(keys)):
            s_pred_log['pred_'+str(keys[i])] = values[i]
        print('----------------------------------------------------------------')
        print(f'yAcc: {y_acc:.4f} | sAcc: {s_acc:.4f}')
        print('----------------------------------------------------------------')
        # performance_log = {f'y_acc': y_acc, f's_acc': s_acc}
        # wandb.log(s_pred_log)
        # wandb.log(performance_log)
    
    # X_tensor, s_tensor, y_tensor = torch.FloatTensor(testset.X), torch.FloatTensor(np.array(testset.s)), torch.FloatTensor(np.array(testset.y))
    zx_te, zs_te, s_te, y_te = encode_all(args, testset, model, clf_xy, device, is_train=False)
    nearest_k = 3
    manifold = MANIFOLD(real_features=testset.X, fake_features=zx_te)
    score, score_index = manifold.rarity(k=nearest_k)
    print("======================================================================")
    print("RARITY SCORE")
    print(score[score_index])
    
    return y_acc, s_acc



if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()
    # ------------------- run config
    parser.add_argument("--seed", type=int, default=730, help="one manual random seed")
    parser.add_argument("--run_mode", type=str, default='e2e', choices=["train", "eval", "e2e"])

    # ------------------- flag & name
    parser.add_argument("--data_name", type=str, default='escape', help="Dataset name")
    parser.add_argument("--model_name", type=str, default='lstm', help="Model ID")
    parser.add_argument("--save_name", type=str, default='default', help="specific string or system time at run start(by default)")

    # ------------------- train config
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    # parser.add_argument("--end_fac", type=float, default=0.001, help="linear decay end factor")
    # parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    # parser.add_argument("--clip_val", type=float, default=2.0, help="gradient clipping value for VAE backbone when using DCD loss")
    # parser.add_argument("--early_stop", type=int, default=0, choices=[1, 0])
    # parser.add_argument("--last_epmod", type=int, default=0, choices=[1, 0], help='use last epoch model or best model at the end of 1 stage train')
    # parser.add_argument("--last_epmod_eval", type=int, default=1, choices=[1, 0], help='use last epoch model or best model for evaluation')
    # parser.add_argument("--eval_model", type=str, default='lr', choices=['mlp', 'lr', 'disc'], help='representation quality evaluation model')

    parser.add_argument("--model_file", type=str)

    args = parser.parse_args()
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------------- specific argument setup for each dataset

    args.epochs = 2000
    args.model_path = './model_german'
    args.data_path = './data/german/'
    args.result_path = './result_german'
    
    # -------------------------------------------------------------------

    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.isdir(args.result_path):
        os.mkdir(args.result_path)

    if args.run_mode == 'train':
        main(args, device)
    else args.run_mode == 'eval':
        evaluation(args, device)
    