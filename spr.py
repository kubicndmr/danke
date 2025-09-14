import os
import data
import wandb
import torch
import utils
import model
import losses
import metrics
import argparse
import numpy as np

#################################################################
################ Epoch Train/Validation Functions ###############
#################################################################


def train_epoch(surgical_model, optimizer, train_dataset, criteria,
                error_train, metrics_train, epoch, device, log_txt):

    surgical_model.train()
    for data_loader in train_dataset['data']:
        # OP-wise iter
        for _, embed_, label_ in data_loader:
            # Data
            embed_ = embed_.clone().detach().float().to(device).unsqueeze(-1)
            label_ = label_.clone().detach().long().to(device).squeeze()

            # Model
            optimizer.zero_grad(set_to_none=True)
            predict_ = surgical_model(embed_)

            # Loss
            error = criteria(predict_, label_)
            error_train[epoch] += error.clone().detach()

            # Metrics
            metrics_train.batch(label_, predict_)

            # BP
            error.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                surgical_model.parameters(), max_norm=1.0)

            # Optimizer Update
            optimizer.step()

        # Log OP
        metrics_train.op_end(data_loader.dataset.op_name, False)

    # Scale Error Function
    error_train[epoch] /= train_dataset['batch_size']

    # Print Metrics
    utils.print_log(
        f"""\tLoss [{args.loss}]\t: {error_train[epoch].item():.5f}""",
        log_txt, display=True)
    utils.print_log(
        f"\tLearning Rate\t: {optimizer.param_groups[0]['lr']}", log_txt, display=True)

    # Metrics Log
    metrics_train.epoch_end(epoch, False)


def eval_epoch(surgical_model, valid_dataset, criteria, error_valid, 
               metrics_valid, epoch, device, log_txt, plot_ribbon):

    surgical_model.eval()
    for data_loader in valid_dataset['data']:
        # OP-wise iter
        for _, embed_, label_ in data_loader:
            # Data
            embed_ = embed_.clone().detach().float().to(device).unsqueeze(-1)
            label_ = label_.clone().detach().long().to(device).squeeze()

            # Model
            with torch.no_grad():
                predict_ = surgical_model(embed_)

            # Loss
            error_ce = criteria(predict_, label_)
            error_valid[epoch] += error_ce.clone().detach()

            # Metrics
            metrics_valid.batch(label_, predict_)

        # Log OP
        metrics_valid.op_end(data_loader.dataset.op_name, plot_ribbon)

    # Scale Error Function
    error_valid[epoch] /= valid_dataset['batch_size']

    # Print Metrics
    utils.print_log(
        f"""\tLoss [{args.loss}]\t: {error_valid[epoch].item():.5f}""",
        log_txt, display=True)

    # Metrics Log
    metrics_valid.epoch_end(epoch, True)

#################################################################
####################### Big Run Function ########################
#################################################################


def fit(args):
    ####################
    ## Data and Paths ##
    ####################
    output_dir = utils.output_dir(args)
    wandb.run.name = output_dir[len("logs/"):]

    log_txt = utils.init_log(output_dir)

    utils.print_log('---{ Args }---', log_txt)
    with open(log_txt, "a") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    dataset = utils.data_split(
        args.real_data_path,
        args.syn_data_path,
        args.real_dataset_size,
        args.syn_dataset_size,
        args.eval_ratio,
        args.n_splits,
        log_txt
    )

    syntrainset = data.get_dataset(
        dataset["syntrainset"], batch_size=args.batch_size)
    syntestset = data.get_dataset(
        dataset["syntestset"], batch_size=args.batch_size)

    #########################
    ## Training Parameters ##
    #########################
    params = {
        "patience_limit": 5,
        "epochs_limit": 500,
        "delta_escb": 0,
    }

    ####################
    ## Surgical Model ##
    ####################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surgical_model = model.SLPNet(
        model_dim=args.model_dim,
        num_classes=8,
        sentence_dropout=args.sentence_dropout,
        model_dropout=args.model_dropout
    ).to(device)

    utils.print_log('\n---{ Model }---', log_txt)
    utils.print_log(surgical_model, log_txt)
    trainable_params = sum(p.numel()
                           for p in surgical_model.parameters() if p.requires_grad)
    utils.print_log(
        f'\nNumber of trainable parameters: {trainable_params:,}\n', log_txt)

    ####################
    ## Loss functions ##
    ####################
    phase_weights = utils.phase_weights(
        syntrainset["data"],
        os.path.join(output_dir, f'results/class_dist_syn.jpg')
    ).to(device)
    
    if args.loss == 'WCE':
        criteria = losses.WCELoss(weight=phase_weights)
    elif args.loss == 'Focal':
        criteria = losses.FocalLoss(
            weight=phase_weights,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma
        )
    elif args.loss == 'LDAM':
        phases = data.get_phase_count(dataset["syntrainset"])
        criteria = losses.LDAMLoss(
            cls_num_list=phases, 
            max_m=args.ldam_m, 
            s=args.ldam_s, 
            weight=phase_weights
        )

    utils.print_log('\n---{ Losses }---', log_txt)
    utils.print_log(criteria, log_txt)
    utils.print_log(f"Phase Weights: \n\t{phase_weights}", log_txt)

    ###############
    ## Optimizer ##
    ###############
    optimizer = torch.optim.Adam(
        surgical_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    utils.print_log('\n---{ Optimizer }---', log_txt)
    utils.print_log(optimizer, log_txt)

    ###############
    ## Scheduler ##
    ###############
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=params['patience_limit'],
        min_lr=0
    )

    utils.print_log('\n---{ Scheduler }---', log_txt)
    utils.print_log(scheduler, log_txt)

    #############
    ## Metrics ##
    #############
    metrics_train = metrics.SPRMetrics(
        log_txt, output_dir, params['epochs_limit'])
    error_train = torch.zeros(params['epochs_limit']).to(device)
    metrics_valid = metrics.SPRMetrics(
        log_txt, output_dir, params['epochs_limit'])
    error_valid = torch.zeros(params['epochs_limit']).to(device)

    ###############################################################
    ###################### Start Pretraining! #####################
    ###############################################################
    epoch = 0
    patience = 0
    best_error = np.inf
    plot_ribbon = False
    best_model_state = None
    early_stopper_flag = False

    utils.print_log('\n---{ Training }---', log_txt)
    while ((epoch < params['epochs_limit']) and (early_stopper_flag == False)
           and (syntrainset['batch_size'] != 0)):
        # Train
        utils.print_log(f'\nEpoch [train]: {epoch}', log_txt, display=True)
        train_epoch(surgical_model,
                    optimizer,
                    syntrainset,
                    criteria,
                    error_train,
                    metrics_train,
                    epoch,
                    device,
                    log_txt
                    )

        # Validation
        utils.print_log(f'\nEpoch [valid]: {epoch}', log_txt, display=True)
        eval_epoch(surgical_model,
                   syntestset,
                   criteria,
                   error_valid,
                   metrics_valid,
                   epoch,
                   device,
                   log_txt,
                   plot_ribbon
                   )

        # Scheduler
        last_error = error_valid[epoch].item()
        scheduler.step(last_error)
        utils.print_log(
            f'\tLearning Rate\t: {scheduler.get_last_lr()}', log_txt, display=True)

        # Loss Check
        if last_error < best_error:
            patience = 0
            best_error = last_error
            best_model_state = surgical_model.state_dict()
        else:
            patience += 1

        if patience < params['patience_limit']:
            utils.print_log(
                f'\tPatience\t: {patience}/{params["patience_limit"]} ({last_error:.3f}/{best_error:.3f})',
                log_txt, display=True)
        else:
            early_stopper_flag = True

        # Increment
        epoch += 1

    #######################################
    ## Plot final error/metric functions ##
    #######################################
    if syntrainset['batch_size'] != 0:
        metrics_train.eval_end(f'pretrain_train')
        metrics_valid.eval_end(f'pretrain_validation')

        utils.plot_error(
            error_train,
            error_valid,
            output_dir,
            'pretrain_'
        )

    ###############################################################
    ####################### Start Finetuning! #####################
    ###############################################################
    plot_ribbon = False
    confusion_matrices = []
    results = np.zeros((args.n_splits, len(metrics_train.metric_keys)))
    
    ######################
    ## K-Fold Iteration ##
    ######################
    for fold in range(args.n_splits):
        utils.print_log(f"\n---{{ Finetuning {fold+1}th-Fold }}---", log_txt)

        ##########
        ## Data ##
        ##########
        realtrainset = data.get_dataset(
            dataset["realtrainsets"][fold], batch_size=args.batch_size)
        realtestset = data.get_dataset(
            dataset["realtestsets"][fold], batch_size=args.batch_size)
        assert not set(dataset["realtrainsets"][fold]).intersection(
            dataset["realtestsets"][fold])

        ###############
        #### Model ####
        ###############
        surgical_model = model.SLPNet(
            model_dim=args.model_dim,
            num_classes=8,
            sentence_dropout=args.sentence_dropout,
            model_dropout=args.model_dropout
        ).to(device)
        if best_model_state is not None:
            surgical_model.load_state_dict(best_model_state)
            utils.print_log("Best pretrained model reloaded!", log_txt)

        ####################
        ## Loss functions ##
        ####################
        phase_weights = utils.phase_weights(
            realtrainset["data"],
            os.path.join(
                output_dir, f'results/class_dist_fold{fold+1}_train.jpg')
        ).to(device)

        if args.loss == 'WCE':
            criteria = losses.WCELoss(weight=phase_weights)
        elif args.loss == 'Focal':
            criteria = losses.FocalLoss(
                weight=phase_weights,
                alpha=args.focal_alpha,
                gamma=args.focal_gamma
            )
        elif args.loss == 'LDAM':
            phases = data.get_phase_count(dataset["syntrainset"])
            criteria = losses.LDAMLoss(
                cls_num_list=phases, 
                max_m=args.ldam_m, 
                s=args.ldam_s, 
                weight=phase_weights
            )

        utils.print_log('\n---{ Losses }---', log_txt)
        utils.print_log(criteria, log_txt)
        utils.print_log(f"Phase Weights: \n\t{phase_weights}", log_txt)

        ###############
        ## Optimizer ##
        ###############
        optimizer = torch.optim.Adam(
            surgical_model.parameters(),
            lr=args.learning_rate * 0.1,
            weight_decay=args.weight_decay
        )

        utils.print_log('\n---{ Optimizer }---', log_txt)
        utils.print_log(optimizer, log_txt)

        #############
        ## Metrics ##
        #############
        metrics_train_ft = metrics.SPRMetrics(
            log_txt, output_dir, params['epochs_limit'])
        error_train_ft = torch.zeros(params['epochs_limit']).to(device)
        metrics_valid_ft = metrics.SPRMetrics(
            log_txt, output_dir, params['epochs_limit'])
        error_valid_ft = torch.zeros(params['epochs_limit']).to(device)

        epoch = 0
        patience = 0
        best_error = 0
        early_stopper_flag = False

        while (epoch < params['epochs_limit']) and (early_stopper_flag == False):
            # Train
            utils.print_log(f'\nEpoch [train]: {epoch}', log_txt, display=True)
            train_epoch(surgical_model,
                        optimizer,
                        realtrainset,
                        criteria,
                        error_train_ft,
                        metrics_train_ft,
                        epoch,
                        device,
                        log_txt
                        )

            # Validation
            utils.print_log(f'\nEpoch [valid]: {epoch}', log_txt, display=True)
            eval_epoch(surgical_model,
                       realtestset,
                       criteria,
                       error_valid_ft,
                       metrics_valid_ft,
                       epoch,
                       device,
                       log_txt,
                       plot_ribbon
                       )
            
            # Scheduler
            last_error = error_valid_ft[epoch].item()
            scheduler.step(last_error)
            utils.print_log(
                f'\tLearning Rate\t: {scheduler.get_last_lr()}', log_txt, display=True)

            # Loss Check
            if last_error > best_error:
                patience = 0
                best_error = last_error
            else:
                patience += 1

            if patience < params['patience_limit']:
                utils.print_log(
                    f'\tPatience\t: {patience}/{params["patience_limit"]} ({last_error:.3f}/{best_error:.3f})',
                    log_txt, display=True)
            else:
                early_stopper_flag = True

            # Increment
            epoch += 1

        #######################################
        ## Plot final error/metric functions ##
        #######################################
        metrics_train_ft.eval_end(f"finetune_{fold+1}_train")
        results[fold, :], confusion_matrices_fold = metrics_valid_ft.eval_end(
            f"finetune_{fold+1}_validation")
        confusion_matrices.extend(confusion_matrices_fold)

        utils.plot_error(error_train_ft,
                         error_valid_ft,
                         output_dir,
                         f"finetune_{fold+1}_"
                         )

    # Log Average Results
    utils.plot_confusion_matrix(confusion_matrices, output_dir)

    results_std = np.std(results, axis=0)
    results_mean = np.mean(results, axis=0)
    for idx, metric in enumerate(metrics_valid_ft.metric_keys):
        utils.print_log(f"[Average {metric}]\t: {results_mean[idx]:.5f} +- {results_std[idx]:.5f}",
                        log_txt,
                        display=True)
        wandb.log({metric: results_mean[idx]})

    ################
    ## Save model ##
    ################
    """
    torch.save({
        'model_state_dict': surgical_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_dir + 'checkpoint.ckp')
    """


#################################################################
############################# Main ##############################
#################################################################
if __name__ == '__main__':
    ## Args ##
    parser = argparse.ArgumentParser(
        description="Train a network for SPR")

    parser.add_argument('--learning_rate',
                        type=float,
                        help='initial learning rate for optimizer')

    parser.add_argument('--weight_decay',
                        type=float,
                        help='regularizer of the optimizer')

    parser.add_argument('--sentence_dropout',
                        type=float,
                        help='dropout probability')

    parser.add_argument('--model_dropout',
                        type=float,
                        help='dropout probability')

    parser.add_argument('--model_dim',
                        type=int,
                        help='embedding dimension of the model')

    parser.add_argument('--real_data_path',
                        type=str, default="/DATA/kubi/Dataset/PoCaP-large-v3/",
                        help='path to real dataset')

    parser.add_argument('--syn_data_path',
                        type=str, default="/DATA/kubi/Dataset/SynPoCaP/",
                        help='path to synthetic dataset')

    parser.add_argument('--real_dataset_size',
                        type=int, default=38,
                        help='number of operations to include in finetuning')

    parser.add_argument('--syn_dataset_size',
                        type=int, default=0,
                        help='number of operations to include in pretraining')

    parser.add_argument('--eval_ratio',
                        type=float, default=0.25,
                        help='portion of the dataset used for testing')

    parser.add_argument('--n_splits',
                        type=int, default=4,
                        help='number of folds')

    parser.add_argument('--batch_size',
                        type=int, default=512,
                        help='number of sentences in the batch')

    parser.add_argument('--loss',
                        type=str, default='WCE',
                        choices=['WCE', 'Focal', 'LDAM'],
                        help='Error function')

    parser.add_argument('--focal_alpha',
                        type=float, default=0.25,
                        help='alpha parameter of Focal Loss')

    parser.add_argument('--focal_gamma',
                        type=float, default=2,
                        help='alpha parameter of Focal Loss')
    
    parser.add_argument('--ldam_m',
                        type=float, default=0.5,
                        help='max m parameter of LDAM Loss')

    parser.add_argument('--ldam_s',
                        type=float, default=30,
                        help='S parameter of LDAM Loss')

    args = parser.parse_args()

    wandb.init()

    fit(args)
