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

    # Init output folder
    output_path, log_txt = utils.init_log(args)

    # wandb
    wandb.init(name=output_path)
    wandb.config.update(args)

    # Data
    data_path = {'train': args.trainset_path,
                 'valid': args.validset_path, 'test': args.testset_path}

    trainset, validset, testset = data.get_dataset(data_path,
                                                   log_txt,
                                                   args.num_ops,
                                                   args.batch_size
                                                   )
    num_classes = 9

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        utils.print_log(
            "\n\nEpoch\t: {}/{}".format(epoch+1, args.epochs),
            log_txt,
            display=True
        )

        print('\nTraining...')
        surgical_model.train()
        for i, data_loader in enumerate(trainset):
            print("\t\tEpoch progress: {:.2f} %".format(
                (i+1)/len(trainset)*100), end='\r')

            # print(data_loader.dataset.op_name)
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
        utils.print_log(
            f"\tValidation Loss\t: {error_valid[epoch].item()}", log_txt, display=True)
        wandb.log(
            {"error_train": error_train[epoch], "error_valid": error_valid[epoch]})
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

    # Log memory usage
    utils.print_log(torch.cuda.memory_summary(device=device), log_txt)
