def train_val(args, config=None):

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        
        # Get Datasets and Loaders
        train_dataset, valid_dataset, train_loader, val_loader = build_dataset(config.batch_size,
                                                                               data_file=args.data_file,
                                                                               files_path=args.label_file)
                                                                               
        # Instantiate the model and Freeze(when True) the basemodel/pretrained ImageNet-weights/biases!
        model = helper.load_model(hidden_layers=config.hidden_layers,
                                  drop_prob=config.drop_prob,
                                  freeze=True, 
                                  num_class=train_dataset.num_classes)
                                                                                      
        model = model.to(device)
        wandb.watch(model, log_freq=50)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
    
        train_loss_list, val_loss_list, val_acc_list = [], [], []
        val_loss_prev = 100.

        for e in range(config.epochs):
            start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            train_loss_list.append(train_loss)
            val_dic, val_loss = val_epoch(model, val_loader, criterion)
            val_acc = val_dic['accuracy_score']
            print(val_dic)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            stats = 'Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f, Val Accuracy: %.2f, Time per epoch: %d' % (e+1, 
                                                                                                                 config.epochs, 
                                                                                                                 train_loss, 
                                                                                                                 val_loss, 
                                                                                                                 val_acc, 
                                                                                                                 (time.time() - start))
            print(stats)
            
            # Save the model with the best val_loss ever/overwrite the old checkpoint
            if val_loss < val_loss_prev:
                val_loss_prev = val_loss

                checkpoint = {'dataset': config.dataset,
                  'model': config.basemodel,
                  'val_loss': val_loss_prev,
                  'state_dict': model.state_dict(),
                  'hidden_layers': config.hidden_layers,
                  'learning_rate': config.learning_rate,
                  'drop_prob': config.drop_prob,
                  'num_class': len(train_dataset.class_names)
                  }
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, '{}_{}.pth'.format(checkpoint['model'],
                                                                                         wandb.run.name
                                                                                                  )))
            else:
                pass

            
            # WandB-logs=Losses and accuracy_score (Most important ones) per epoch:
            wandb.log({'train_loss': train_loss, 
                       'val_loss': val_loss, 
                       'val_acc': val_acc
                      })
            
            # Logs from the val_dic
            for metric in val_dic:
                # if metric is the array for the auc_roc of each class
                if metric == 'roc_auc_score':
                    for i,score in enumerate(val_dic[metric]):
                        wandb.log({(metric+'/'+train_dataset.class_names[i]): score})
                else:
                    wandb.log({metric: val_dic[metric]})
            
        # 'best' Logs of the whole run/sweep:
        best_train_loss = min(enumerate(train_loss_list), key=itemgetter(1))
        best_val_loss = min(enumerate(val_loss_list), key=itemgetter(1))
        best_val_acc = min(enumerate(val_acc_list), key=itemgetter(1))
        wandb.log({'best_train_loss': best_train_loss[1], 'best_train_epoch': best_train_loss[0],
                   'best_val_loss': best_val_loss[1], 'best_val_epoch': best_val_loss[0],
                   'best_val_acc': best_val_acc[1], 'best_acc_epoch': best_val_acc[0]
                  })
        print('Result list: ', train_loss_list, val_loss_list)
        print('Best train_epoch: ', best_train_loss[0],'Best train_loss: ', best_train_loss[1])
        print('Best val_epoch: ', best_val_loss[0], 'Best val_loss: ', best_val_loss[1])
        print('Best acc_epoch: ', best_val_acc[0], 'Best val_acc: ', best_val_acc[1])



def main():
    

    # Training settings
    parser = argparse.ArgumentParser(description='Hyperparametertuning for Multi-Target Classification')
    parser.add_argument('--data_dir', type=str, default='image_folder', metavar='N',
                        help='input directory of the dataset and label-file (default: image_folder)')
    parser.add_argument('--label_file', type=str, default='labels.csv', metavar='N',
                        help='Name of the label-file (default: labels.csv)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', metavar='N',
                        help='Name of the checkpoint-directory (default: checkpoints)')
    parser.add_argument('--config_file', type=str, default='sweep-bayes-hyperband.yaml', metavar='N',
                        help='Path of the config-file (default: sweep-bayes-hyperband.yaml)')
    parser.add_argument('--project', type=str, default='Multi-Target-Classifier', metavar='N',
                        help='To which project on WandB do you want to log the data (default: Multi-Target-Classifier)')
    parser.add_argument('--entity', type=str, default='Group', metavar='N',
                        help='To which group on WandB do you want to log the data (default: Group)')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 666)')
    parser.add_argument('--sweeps', type=int, default=10, metavar='S',
                        help='Amount of sweeps (default: 10)')
    parser.add_argument("-q","--dry_run", action="store_true", help="Dry run (do not log to wandb)")  

    args = parser.parse_args()

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    
    # Set the seeds for reproducibility to the number of the beast:
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    # Initialize sweep:
    wandb.init(config=args.config_file, project=args.project)
    sweep_config = wandb.config
    sweep_id = wandb.sweep(sweep=sweep_config, entity=args.entity, project=args.project)

    wandb.agent(sweep_id, train_val(args), count=args.sweeps)
        

if __name__ == '__main__':
    main()