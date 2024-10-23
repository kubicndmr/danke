import os
import data
import wandb
import utils
import torch
import model
import metrics
import argparse
import numpy as np

if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(
        description="Train a network for SPR")

    parser.add_argument('-p', '--project_name', type=str,
                        help='name of training')

    parser.add_argument('-t', '--training_tag', type=str,
                        help='identification tag of training')

    parser.add_argument('-e', '--epochs',
                        type=int, default=1000,
                        help='number of epochs to train')

    parser.add_argument('-b', '--batch_size',
                        type=int, default=64,
                        help='batch size of training data')

    parser.add_argument('-m', '--model_dropout',
                        type=float, default=0,
                        help='probability of dropping neural connection')

    parser.add_argument('-s', '--sentence_dropout',
                        type=float, default=0.25,
                        help='probability of dropping neural connection')

    parser.add_argument('-l', '--learning_rate',
                        type=float, default=1e-5,
                        help='initial learning rate for optimizer')

    parser.add_argument('-w', '--weight_decay',
                        type=float, default=1e-6,
                        help='regularizer of optimizer')

    parser.add_argument('-d', '--model_dim',
                        type=int, default=256,
                        help='hidden vector size of model')

    parser.add_argument('-r', '--repeat_run',
                        type=int, default=5,
                        help='Number of times repeat training and average results')

    parser.add_argument('-no', '--num_ops',
                        type=int, default=-1,
                        help='Number of OPs to use in training')

    parser.add_argument('-tr', '--trainset_path', type=str,
                        help='path to training set')

    parser.add_argument('-va', '--validset_path', type=str,
                        help='path to validation set')

    parser.add_argument('-te', '--testset_path', type=str,
                        help='path to test set')

    args = parser.parse_args()

    # wandb init
    wandb.init()
    wandb.config.update(args)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Average results over runs
    train_loss_container = np.zeros((args.repeat_run, args.epochs))
    valid_loss_container = np.zeros((args.repeat_run, args.epochs))
    accuracy_container = np.zeros((args.repeat_run, args.epochs))
    f1_container = np.zeros((args.repeat_run, args.epochs))
    jaccard_container = np.zeros((args.repeat_run, args.epochs))

    for run in range(args.repeat_run):
        # Init output folder
        output_path, log_txt = utils.init_log(args, run)

        # Data
        data_path = {'train': args.trainset_path,
                     'valid': args.validset_path,
                     'test': args.testset_path
                     }

        trainset, validset, testset = data.get_dataset(data_path,
                                                       log_txt,
                                                       args.num_ops,
                                                       args.batch_size
                                                       )
        num_classes = 9

        # Initialize model for each run
        surgical_model = model.SLPNet(
            model_dim=args.model_dim,
            num_classes=num_classes,
            model_dropout=args.model_dropout,
            sentence_dropout=args.sentence_dropout
        ).to(device)

        utils.print_log('\n---{ Model }---', log_txt)
        utils.print_log(surgical_model, log_txt)
        utils.print_log('\nNumber of Parameters: {:,}\n'.format(sum(p.numel()
                        for p in surgical_model.parameters() if p.requires_grad)),
                        log_txt)

        # Loss function
        phase_weights = utils.phase_weights(trainset).to(device)
        criteria = torch.nn.CrossEntropyLoss(
            weight=phase_weights, reduction='mean', ignore_index=8)

        # Optimizer
        optimizer = torch.optim.Adam(
            surgical_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Training
        metrics_train = metrics.SPRMetrics(log_txt, output_path, args.epochs)
        metrics_valid = metrics.SPRMetrics(log_txt, output_path, args.epochs)
        error_train = torch.zeros(args.epochs).to(device)
        error_valid = torch.zeros(args.epochs).to(device)
        early_stopper_flag = False
        patience_limit = 5
        patience_escb = 0
        delta_escb = 0.001
        best_loss = 1E9
        epoch = 0

        trainset_size = np.sum([d_l.dataset.__len__() for d_l in trainset])
        validset_size = np.sum([d_l.dataset.__len__() for d_l in validset])

        # Iter epochs
        while (epoch < args.epochs) and (early_stopper_flag == False):
            print(f"--- Run {run + 1} ---")
            utils.print_log(
                "Epoch\t: {}/{}".format(epoch+1, args.epochs),
                log_txt,
                display=True
            )

            print('\nTraining...')
            surgical_model.train()
            for i, data_loader in enumerate(trainset):
                print("\t\tEpoch progress: {:.2f} %".format(
                    (i+1)/len(trainset)*100), end='\r')

                # OP-wise
                for time_, embed_, label_ in data_loader:
                    # Data
                    time_ = time_.clone().detach().float().to(device)
                    embed_ = embed_.clone().detach().float().to(device).unsqueeze(-1)
                    label_ = label_.clone().detach().long().to(device).squeeze()

                    # Model
                    optimizer.zero_grad()
                    predict_ = surgical_model(embed_, time_)

                    # Loss
                    error_batch = criteria(predict_, label_)
                    error_train[epoch] += error_batch

                    # Metrics
                    metrics_train.batch(label_, predict_)

                    # BP
                    error_batch.backward()
                    optimizer.step()

                # Log OP
                metrics_train.op_end(
                    data_loader.dataset.op_name, plot_ribbon=False)

            # Train Log
            error_train[epoch] /= trainset_size
            train_loss_container[run, epoch] += error_train[epoch].item()
            utils.print_log(
                f"\tTrain Loss\t: {error_train[epoch].item()}", log_txt, display=True)
            metrics_train.epoch_end(epoch)

            print('\nValidating...')
            surgical_model.eval()
            for i, data_loader in enumerate(validset):
                print("\t\tEpoch progress: {:.2f} %".format(
                    (i+1)/len(validset)*100), end='\r')

                # OP-wise
                for time_, embed_, label_ in data_loader:
                    # Data
                    time_ = time_.clone().detach().float().to(device)
                    embed_ = embed_.clone().detach().float().to(device).unsqueeze(-1)
                    label_ = label_.clone().detach().long().to(device).squeeze()

                    # Model
                    with torch.no_grad():
                        predict_ = surgical_model(embed_, time_)

                    # Loss
                    error_batch = criteria(predict_, label_)
                    error_valid[epoch] += error_batch

                    # Metrics
                    metrics_valid.batch(label_, predict_)

                # Log OP
                metrics_valid.op_end(
                    data_loader.dataset.op_name, plot_ribbon=False)

            # Validation Log
            error_valid[epoch] /= validset_size
            valid_loss_container[run, epoch] += error_valid[epoch].item()
            utils.print_log(
                f"\tValidation Loss\t: {error_valid[epoch].item()}", log_txt, display=True)
            metrics_valid.epoch_end(epoch)

            # Early Stopper
            if error_valid[epoch] < best_loss:
                best_loss = error_valid[epoch]
                patience_escb = 0
                # torch.save({
                #    'epoch': epoch + 1,
                #    'model_state_dict': surgical_model.state_dict(),
                #    'optimizer_state_dict': optimizer.state_dict(),
                #    'loss': best_loss,
                # }, output_path + '/results/checkpoint.ckp')
            if error_valid[epoch] > best_loss + delta_escb:
                patience_escb += 1
            if patience_escb > patience_limit:
                early_stopper_flag = True
                utils.print_log('Early Stopper!!!\n', log_txt, display=True)

            epoch += 1

        # Log
        metrics_train.eval_end('train')
        metrics_valid.eval_end('validation')
        utils.plot_error(error_train, error_valid, output_path)
        accuracy_container[run, :] = metrics_valid.real_metrics[:, 0]
        f1_container[run, :] = metrics_valid.real_metrics[:, 1]
        jaccard_container[run, :] = metrics_valid.real_metrics[:, 4]

        # Log memory usage
        utils.print_log(torch.cuda.memory_summary(device=device), log_txt)


    # Log averaged results to W&B
    best_train_losses = []
    best_valid_losses = []
    best_accuracies = []
    best_f1s = []
    best_jaccards = []

    for run in range(args.repeat_run):
        train_loss_run = utils.remove_tailzeros(train_loss_container[run, :])
        valid_loss_run = utils.remove_tailzeros(valid_loss_container[run, :])
        
        best_train_loss = np.min(train_loss_run)
        best_valid_loss = np.min(valid_loss_run)
        best_accuracy = np.max(accuracy_container[run, :])
        best_f1 = np.max(f1_container[run, :])
        best_jaccard = np.max(jaccard_container[run, :])
        
        best_train_losses.append(best_train_loss)
        best_valid_losses.append(best_valid_loss)
        best_accuracies.append(best_accuracy)
        best_f1s.append(best_f1)
        best_jaccards.append(best_jaccard)

    # Log averaged best results across all runs
    wandb.log({
        "avg_best_train_loss": np.mean(best_train_losses),
        "avg_best_valid_loss": np.mean(best_valid_losses),
        "avg_best_accuracy": np.mean(best_accuracies),
        "avg_best_f1": np.mean(best_f1s),
        "avg_best_jaccard": np.mean(best_jaccards)
    })


    output_path = os.path.dirname(output_path[:-1])+'/'
    os.mkdir(output_path+'results')
    utils.plot_error(train_loss_container.T, valid_loss_container.T, output_path)
    utils.print_log(torch.cuda.memory_summary(device=device), log_txt)
