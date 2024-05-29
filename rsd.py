'''
Remaining surgery duration estimation 
via sentences in pocap dataset
'''
import sys
import data
import wandb
import utils
import torch
import model
import argparse
import numpy as np

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
wandb.login()
      
if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(
        description="Train a network for RSD")
    
    parser.add_argument('-p', '--project_name', type=str,
                        help='name of training')
    
    parser.add_argument('-t', '--training_tag', type=str,
                        help='identification tag of training')
    
    parser.add_argument('-e', '--epochs', 
                        type=int, default=100,
                        help='number of epochs to train')
    
    parser.add_argument('-b', '--batch_size', 
                        type=int, default=16,
                        help='batch size of training data')
    
    parser.add_argument('-d', '--dropout_prob', 
                        type=float, default=0,
                        help='probability of dropping neural connection')
    
    parser.add_argument('-l', '--learning_rate', 
                        type=float, default=1e-5,
                        help='initial learning rate for optimizer')
    
    parser.add_argument('-w', '--weight_decay', 
                        type=float, default=1e-6,
                        help='regularizer of optimizer')
    
    parser.add_argument('-n', '--model_dim', 
                        type=int, default=256,
                        help='hidden vector size of model')
    
    parser.add_argument('-nh', '--num_head', 
                        type=int, default=4,
                        help='number of heads in MHA')
    
    parser.add_argument('-ne', '--num_enc', 
                        type=int, default=3,
                        help='number of stacked encoders')
    
    args = parser.parse_args()
    
    # wandb
    wandb.init(project=args.project_name)
    wandb.config.update(args)
    
    # Init output folder 
    output_path, log_txt = utils.init_log(args)
    
    # Get data
    data_split = utils.data_split('Data/', log_txt)
    trainset, validset, testset = data.get_dataset('Data/annotations.json',
                                                   data_split,
                                                   args.model_dim,
                                                   args.batch_size)
    
    
    # Init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surgical_model = model.SLPNet(
        model_dim=args.model_dim,
        num_head=args.num_head,
        num_encoder=args.num_enc,
        num_classes=1,
        dropout_prob=args.dropout_prob
    ).to(device)
    utils.print_log('Number of Parameters: {:,}'.format(sum(p.numel() 
                    for p in surgical_model.parameters() if p.requires_grad)), 
                    log_txt)
    
    # Loss function
    criteria = torch.nn.MSELoss(reduction='mean')
    
    # Optimizer 
    optimizer = torch.optim.Adam(
        surgical_model.parameters(), 
        lr = args.learning_rate, 
        weight_decay = args.weight_decay
    )
    
    ## Train
    error_train = torch.zeros(args.epochs).to(device)
    error_valid = torch.zeros(args.epochs).to(device)
    early_stopper = False
    patience_limit = 5
    patience_escb = 0
    delta_escb = 0.1
    best_loss = 1E9
    epoch = 0
    
    while (epoch < args.epochs) and (early_stopper == False):
        utils.print_log(
            "\n\nEpoch\t: {}/{}".format(epoch+1, args.epochs), 
            log_txt,
            display=True
        )
        
        print('\nTraining...')
        for i, data_loader in enumerate(trainset):
            print("\t\tEpoch progress: {:.2f} %".format((i+1)/len(trainset)*100), 
                  end = '\r')
            
            surgical_model.train()
            for (time_, embed_, _, label_) in data_loader:
                time_ = time_.clone().detach().float().to(device)
                embed_ = embed_.clone().detach().float().to(device)
                label_ = label_.clone().detach().float().to(device)
                
                optimizer.zero_grad()
                
                predict_ = surgical_model(embed_, time_)
                
                error_batch = criteria(predict_, label_)
                error_train[epoch] += error_batch 
                
                error_batch.backward()
                optimizer.step()
        
        print('\nValidating...')
        for i, data_loader in enumerate(validset):
            print("\t\tEpoch progress: {:.2f} %".format((i+1)/len(validset)*100), 
                  end = '\r')
            
            surgical_model.eval()
            for (time_, embed_, _, label_) in data_loader:
                time_ = time_.clone().detach().float().to(device)
                embed_ = embed_.clone().detach().float().to(device)
                label_ = label_.clone().detach().float().to(device)
                
                with torch.no_grad():
                    predict_ = surgical_model(embed_, time_)
                
                error_batch = criteria(predict_, label_)
                error_valid[epoch] += error_batch
        
        wandb.log({"error_valid": error_valid[epoch], "epoch": epoch})
              
        # early stopper
        if error_valid[epoch] < best_loss:
            best_loss = error_valid[epoch]
            patience_escb = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': surgical_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, output_path + '/results/checkpoint.ckp')
        if error_valid[epoch] > best_loss + delta_escb:    
            patience_escb += 1
        if patience_escb > patience_limit:
            early_stopper = True
            
        epoch += 1

    utils.plot_error(error_train, error_valid, output_path)
    
    
    ## Result from best model
    checkpoint = torch.load(output_path + '/results/checkpoint.ckp')
    surgical_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print('\n\nGenerating results from epoch {}...'.format(checkpoint['epoch']))
    
    for i, data_loader in enumerate(validset):
        op_name = data_loader.dataset.get_op_name()
        print(f'\t{op_name} in validation set')
        
        predicted_rsd = torch.zeros(len(data_loader.dataset))
        true_rsd = torch.zeros(len(data_loader.dataset))
        
        surgical_model.eval()
        for n, (time_, embed_, _, label_) in enumerate(data_loader):
            time_ = time_.clone().detach().float().to(device)
            embed_ = embed_.clone().detach().float().to(device)
            label_ = label_.clone().detach().float().to(device)
            
            with torch.no_grad():
                predict_ = surgical_model(embed_, time_)
                
            predicted_rsd[n*args.batch_size:(n+1)*args.batch_size] = predict_
            true_rsd[n*args.batch_size:(n+1)*args.batch_size] = label_
            
        utils.plot_rsd(predicted_rsd, true_rsd, output_path, op_name)