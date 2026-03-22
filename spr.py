import os
import dotenv
dotenv.load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")

import torch
import argparse
import numpy as np
import huggingface_hub
import SurgPhaseRecog.data as data
import SurgPhaseRecog.utils as utils
import SurgPhaseRecog.model as model
import SurgPhaseRecog.losses as losses
import SurgPhaseRecog.config as config
import SurgPhaseRecog.metrics as metrics


#################################################################
################ Epoch Train/Validation Functions ###############
#################################################################


def train_epoch(surgical_model, optimizer, train_dataset, criteria,
                error_train, metrics_train, epoch, device, log_txt):

    surgical_model.train()
    for data_loader in train_dataset['data']:
        # OP-wise iter
        for token_ids, att_mask, label in data_loader:

            # Data
            token_ids = token_ids.clone().detach().to(device)
            att_mask = att_mask.clone().detach().to(device)
            label = label.clone().detach().long().to(device)

            # Model
            optimizer.zero_grad(set_to_none=True)
            predicted = surgical_model(token_ids, att_mask)

            # Loss
            error = criteria(predicted, label)
            error_train[epoch] += error.clone().detach()

            # Metrics
            metrics_train.batch(label, predicted)

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
        f"""\tLoss\t: {error_train[epoch].item():.5f}""",
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
        for token_ids, att_mask, label in data_loader:
            # Data
            token_ids = token_ids.clone().detach().to(device)
            att_mask = att_mask.clone().detach().to(device)
            label = label.clone().detach().long().to(device)

            # Model
            with torch.no_grad():
                predicted = surgical_model(token_ids, att_mask)

            # Loss
            error_ce = criteria(predicted, label)
            error_valid[epoch] += error_ce.clone().detach()

            # Metrics
            metrics_valid.batch(label, predicted)

        # Log OP
        metrics_valid.op_end(data_loader.dataset.op_name, plot_ribbon)

    # Scale Error Function
    error_valid[epoch] /= valid_dataset['batch_size']

    # Print Metrics
    utils.print_log(
        f"""\tLoss\t: {error_valid[epoch].item():.5f}""",
        log_txt, display=True)

    # Metrics Log
    metrics_valid.epoch_end(epoch, True)

#################################################################
######################### Fit Function ##########################
#################################################################


def fit(args):
    ####################
    ## Data and Paths ##
    ####################
    hparams = utils.sample_hparams()
    configurator = config.SurgConfig(hparams)

    output_dir = utils.output_dir(args, configurator)

    log_txt = utils.init_log(output_dir)

    utils.print_log('---{ Args }---', log_txt)
    with open(log_txt, "a") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        for section_name, section_dict in configurator.__dict__.items():
            for arg, value in section_dict.items():
                f.write(f"{section_name}.{arg}: {value}\n")

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
        dataset["syntrainset"], configurator.dataset_config)
    syntestset = data.get_dataset(
        dataset["syntestset"], configurator.dataset_config)

    ####################
    ## Surgical Model ##
    ####################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surgical_model = model.SLPNet(config=configurator).to(device)

    utils.print_log('\n---{ Model }---', log_txt)
    utils.print_trainable_layers(surgical_model, log_txt)

    ####################
    ## Loss functions ##
    ####################
    phase_weights = utils.phase_weights(
        syntrainset["data"],
        os.path.join(output_dir, f'results/class_dist_syn.jpg')
    ).to(device)
    phase_counts = syntrainset['phase_counts']

    criteria = losses.get_loss_function(
        configurator.loss_config, phase_weights, phase_counts)

    utils.print_log('\n---{ Losses }---', log_txt)
    utils.print_log(criteria, log_txt)
    utils.print_log(f"Phase Weights: \n\t{phase_weights}", log_txt)

    ###############
    ## Optimizer ##
    ###############
    optimizer = torch.optim.Adam(
        surgical_model.parameters(),
        lr=configurator.params["lr"],
        weight_decay=configurator.params["weight_decay"]
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
        patience=configurator.params['patience_limit'],
        min_lr=0
    )

    utils.print_log('\n---{ Scheduler }---', log_txt)
    utils.print_log(scheduler, log_txt)

    #############
    ## Metrics ##
    #############
    metrics_train = metrics.SPRMetrics(
        log_txt, output_dir, configurator.params['epochs_limit'])
    error_train = torch.zeros(configurator.params['epochs_limit']).to(device)
    metrics_valid = metrics.SPRMetrics(
        log_txt, output_dir, configurator.params['epochs_limit'])
    error_valid = torch.zeros(configurator.params['epochs_limit']).to(device)

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
    while (epoch < configurator.params['epochs_limit'] and early_stopper_flag == False
           and syntrainset['batch_size'] != 0):
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

        if patience < configurator.params['patience_limit']:
            utils.print_log(
                f'\tPatience\t: {patience}/{configurator.params["patience_limit"]} ({last_error:.3f}/{best_error:.3f})',
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
    if args.n_splits:
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
                dataset["realtrainsets"][fold], batch_size=configurator.params["batch_size"])
            realtestset = data.get_dataset(
                dataset["realtestsets"][fold], batch_size=configurator.params["batch_size"])
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

            criteria = losses.get_loss_function(
                configurator.loss_config, phase_weights)

            utils.print_log('\n---{ Losses }---', log_txt)
            utils.print_log(criteria, log_txt)
            utils.print_log(f"Phase Weights: \n\t{phase_weights}", log_txt)

            ###############
            ## Optimizer ##
            ###############
            optimizer = torch.optim.Adam(
                surgical_model.parameters(),
                lr=configurator.params["lr"]*0.1,
                weight_decay=configurator.params["weight_decay"]*0.1
            )

            utils.print_log('\n---{ Optimizer }---', log_txt)
            utils.print_log(optimizer, log_txt)

            #############
            ## Metrics ##
            #############
            metrics_train_ft = metrics.SPRMetrics(
                log_txt, output_dir, configurator.params['epochs_limit'])
            error_train_ft = torch.zeros(
                configurator.params['epochs_limit']).to(device)
            metrics_valid_ft = metrics.SPRMetrics(
                log_txt, output_dir, configurator.params['epochs_limit'])
            error_valid_ft = torch.zeros(
                configurator.params['epochs_limit']).to(device)

            epoch = 0
            patience = 0
            best_error = 0
            early_stopper_flag = False

            while (epoch < configurator.params['epochs_limit']) and (early_stopper_flag == False):
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

                if patience < configurator.params['patience_limit']:
                    utils.print_log(
                        f'\tPatience\t: {patience}/{configurator.params["patience_limit"]} ({last_error:.3f}/{best_error:.3f})',
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
                        type=int, default=None,
                        help='number of folds')

    args = parser.parse_args()

    huggingface_hub.login(token=os.getenv("HF_TOKEN"))

    fit(args)
