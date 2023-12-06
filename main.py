import torch
from torch.utils.data import DataLoader
import torchvision
import os
import numpy as np
import time
from tqdm import tqdm
import glob
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from data_module import dataset as ds
from functools import reduce

from timm.utils import NativeScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import wandb
torch.backends.cudnn.benchmark = True
import sys
import platform
from ptflops import get_model_complexity_info

sys.setrecursionlimit(1000)

def main(config):
    # set seed
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    experiment_dir = config["experiment_dir"]  + config["name"]
    if not config["debug"]:
        run = wandb.init(project='natural_image_denoising', group=config['name'], config=config, resume=config["resume"])
        group_id = config['name'] + "_" + config['fourier_mode'] + '_' + str(config['model']['tokenization']) + '_' + str(config['model']["patching_mode"]) + '_' + str(config['model']['token_size']) + '_' + config['normalization_mode'] + "_" + config['criterion_name'] + '_' + str(config['model']['emb_size'])
        config = wandb.config       
        experiment_dir = config["experiment_dir"] + "/" + config['name'] + "/" + run.name

    elif config["debug"]:
        torch.autograd.set_detect_anomaly(True)  # todo: Error MmBackward on Server c1 with GFNet
        # torch.autograd.profiler.profile(True)
        # torch.autograd.profiler.emit_nvtx(True)
        
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    train_ds = ds.DatasetTrain(rgb_dir=)

    test_ds  = DenoisingDataset(root_dir=config["test_dir"],
                                color=config["color"],
                                prefetch=config["prefetch"],
                                real_noise=config["real_noise"],
                                restrict_data=1,
                                transform=transforms.get_torch_transform(mode='val', sigma=config["sigma"], real_noise=config["real_noise"], augment_in_transform=config["augment_in_transform"]))

    train_loader = DataLoader(dataset=train_ds, num_workers=config["num_workers"], drop_last=False, pin_memory=False,
                              batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(dataset=test_ds, num_workers=config["num_workers"], drop_last=False, batch_size=config["batch_size"], shuffle=False)

    print(f"Batches Train: {train_loader.__len__()} , Batches Val: {test_loader.__len__()}")
    print(f"Size of Training Set: {train_ds.__len__()} , Size of Validation Set: {test_ds.__len__()}")

    eda = EDA(train_loader=train_loader, test_loader=test_loader, mode=config["fourier_mode"])
    
    mean_train_noisy, std_train_noisy, mean_train_target, std_train_target = eda.compute_mean_std(config["normalization_mode"])
    # mean_train_noisy, std_train_noisy, mean_train_target, std_train_target = eda.compute_mean_std_kspace()
    print("mean: ", mean_train_noisy, "std: ", std_train_noisy)
    if config["eda"]:
        eda.vis_pixelwise()
        eda.vis_eigenimages()
        eda.vis_histogram()
        eda.vis_pixelwise_kspace()

    # Get the model
        # model_library.get_model(config["experiment"])
    model = model_library.get_model(config["experiment"])(fourier_mode=config["fourier_mode"], **config["model"])
    # model = Uformer(img_size=128,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
    #         depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=3)
    with torch.cuda.amp.autocast():
        try:
            macs, params = get_model_complexity_info(model, (12, 128, 128), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # summary(model, input_size=(1, 6, 256, 129))
        except Exception as e: 
            print("Couldnt estimate complexity")
    device_ids = [i for i in range(torch.cuda.device_count())]
    print("device ids: ", device_ids)
    model = torch.nn.DataParallel(model)
    model.cuda()
    if config["normalization_mode"] != "":
        model.module.data_mean = mean_train_noisy.cuda()
        model.module.data_std = std_train_noisy.cuda()
        model.module.norm_mode = config["normalization_mode"]
    print("Number of params: ", count_parameters(model))

    if not config["debug"]: 
        wandb.watch(model)
 
    # optimizer
    optimizer = model.module.optimizer(config["optimizer_name"])(model.parameters(), **config["optimizer"])
    if not config["warmup_scheduler"]:
        scheduler = model.module.scheduler(config["scheduler_name"])(optimizer, **config["scheduler"])
        scheduler.step()
    else:  # todo: how to avoid to use if-else?
        print("Using warmup strategy!")
        scheduler2 = model.module.scheduler_warmup2(config["warmup_scheduler"]["scheduler_2_name"])(optimizer, **config["warmup_scheduler"]["scheduler_2"])
        scheduler = model.module.scheduler_warmup1(config["warmup_scheduler"]["scheduler_1_name"])(optimizer, **config["warmup_scheduler"]["scheduler_1"], after_scheduler=scheduler2)
        scheduler.step()

    # define the loss function
    criterion = model.module.criterion(config["criterion_name"])(**config["criterion"]).cuda()

    loss_scaler = NativeScaler()
    # load model
    if config["resume"]:  # todo: need to be debugged, e.g. path, nn.DataParallel, learning rate scheduler
        lof = glob.glob(f'{experiment_dir}/model_*.pth')
        latest_ckpt = max(lof, key=os.path.getctime)
        ckpt = load_checkpoint(model, latest_ckpt)
        initial_epoch = load_start_epoch(latest_ckpt) + 1 
        lr = load_optim(optimizer, latest_ckpt)

        for i in range(1, initial_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        best_psnr = ckpt['best_psnr']
        print(f"Found Checkpoint: {latest_ckpt} with Train PSNR: {ckpt['train_psnr']} Val PSNR: {ckpt['best_psnr']}")
        print(f"Resume Training with Learning Rate {new_lr}")
    else:
        initial_epoch = 0
        best_psnr = 0


    ######### validation ###########
    with torch.no_grad():
        model.eval()
        psnr_dataset = []
        psnr_recon = []
        ssim_dataset = []
        ssim_recon = []
        for ii, (inputs, outputs) in enumerate(tqdm(test_loader)):
            noisy, target = inputs[0].cuda(), outputs[0].cuda()
            with torch.cuda.amp.autocast():
                pp_noise = model.module.preprocess(noisy, fourier_mode=config["fourier_mode"])
                recon = model(pp_noise)
                recon = model.module.postprocess(recon, fourier_mode=config["fourier_mode"])
                recon = torch.clamp(recon,0,1)
            psnr_dataset.append(batch_PSNR(noisy, target, True).item())
            psnr_recon.append(batch_PSNR(recon, target, True).item())
            ssim_dataset.append(batch_ssim(noisy, target))
            ssim_recon.append(batch_ssim(recon, target))
        psnr_dataset = sum(psnr_dataset)/len(test_loader)
        psnr_recon = sum(psnr_recon)/len(test_loader)
        ssim_dataset = sum(ssim_dataset)/len(test_loader)
        ssim_recon = sum(ssim_recon)/len(test_loader)
        
        print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_recon))
        print('Input & GT (SSIM) -->%.4f dB'%(ssim_dataset), ', Model_init & GT (SSIM) -->%.4f dB'%(ssim_recon))
    # training loop
    pbar = tqdm(range(initial_epoch, config["num_epochs"]))
    for epoch in pbar:
        ## only DDP
        # train_loader.sampler.set_epoch(epoch)
        # test_loader.sampler.set_epoch(epoch)
        train_loss = 0.0
        train_psnr = 0.0
        model.train()
        cutoff = []

        # value = dropout_scheduler(0.1, 0.0, epoch, config["num_epochs"], mode="original")
        # print("Update Drophead rate to: ", value)
        # set_dropout(model, epoch)

        tin = time.time()
        for sidx, (inputs, outputs) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            tic = time.time()
            noisy, target = inputs[0].cuda(), outputs[0].cuda()
            assert noisy.shape[0] == config["batch_size"], f"Noisy shape differs from the batch size with: noise {noisy.shape}. Either increase your amount of data, or decrease the batch size"

            if epoch > 5:
                target, noisy = transforms.MixUpAug().aug(target, noisy, rank=0)

            with torch.cuda.amp.autocast():
                # todo: consider introducing the loss scaler as in original code or torch.cuda.amp.GradScaler
                # todo: research and discuss with kerstin
                pp_noise = model.module.preprocess(noisy, fourier_mode=config["fourier_mode"])
                recon = model(pp_noise)
                recon = unpack_multistage(recon)
                # print("MAX: ", pp_noise.real.abs().max(), pp_noise.imag.abs().max(), " MIN: ", pp_noise.real.abs().min(), pp_noise.imag.abs().min())
                if recon.shape[-1] == 2:
                    recon = torch.view_as_complex(recon)
                recon = model.module.postprocess(recon, fourier_mode=config["fourier_mode"])

                if "freq" in config["fourier_mode"]:
                    recon = fourier_transform.select_fft(config["fourier_mode"])(recon)
                    target = fourier_transform.select_fft(config["fourier_mode"])(target)
                
                # print("MAX: ", pp_noise.abs().max(), recon.abs().max(), target.abs().max(), " MIN: ", pp_noise.abs().min(), recon.abs().min(), target.abs().min())
                loss = criterion(recon, target) #/ config.batch_size
                print("Loss: ", loss)
            loss_scaler(loss, optimizer, parameters=model.parameters()) #, clip_grad=100, clip_mode='value')
            
            # loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            # optimizer.step()
            train_loss += loss.item() / len(train_loader)

            if "freq" in config["fourier_mode"]:
                recon = fourier_transform.select_ifft(config["fourier_mode"])(recon).real
                target = fourier_transform.select_ifft(config["fourier_mode"])(target).real

            train_psnr += psnr_criterion(recon, target).item() / len(train_loader)
            # print(time.time() - tic)

            if not config["debug"] and sidx == train_loader.__len__()//2:
                wandb.log({"Images/train/noisy": wandb.Image(noisy)})
                wandb.log({"Images/train/target": wandb.Image(target)})
                wandb.log({"Images/train/recon": wandb.Image(recon)})

        elapsed_time = time.time() - tin
        
        # eval loop
        test_loss = 0.0
        test_psnr = 0.0
        test_ssim = 0.0

        model.eval()
        for sidx, (inputs, outputs) in enumerate(test_loader):
            with torch.no_grad():
                noisy, target = inputs[0].cuda(), outputs[0].cuda()
                with torch.cuda.amp.autocast():
                    pp_noise = model.module.preprocess(noisy, fourier_mode=config["fourier_mode"])
                    recon = model(pp_noise)
                    recon = unpack_multistage(recon)
                    if recon.shape[-1] == 2:
                        recon = torch.view_as_complex(recon)
                    recon = model.module.postprocess(recon, fourier_mode=config["fourier_mode"])
                
                    if "freq" in config["fourier_mode"]:
                        recon = fourier_transform.select_fft(config["fourier_mode"])(recon)
                        target = fourier_transform.select_fft(config["fourier_mode"])(target)
                    

                loss = criterion(recon, target)
                test_loss += loss.item() / len(test_loader)
                
                if "freq" in config["fourier_mode"]:
                    recon = fourier_transform.select_ifft(config["fourier_mode"])(recon).real
                    target = fourier_transform.select_ifft(config["fourier_mode"])(target).real

                if epoch == (config["num_epochs"] - 1):
                    test_ssim += batch_ssim(recon, target) / len(test_loader)

                test_psnr += psnr_criterion(recon, target).item() / len(test_loader)

                if not config["debug"] and sidx == test_loader.__len__()//2:
                    wandb.log({"Images/val/noisy": wandb.Image(noisy)})
                    wandb.log({"Images/val/target": wandb.Image(target)})
                    wandb.log({"Images/val/recon": wandb.Image(recon)})

        current_lr = optimizer.param_groups[0]['lr']
        log_dict = {
            'train/loss' : train_loss,
            'train/psnr' : train_psnr,
            'val/loss' : test_loss,
            'val/psnr' : test_psnr,
            'val/ssim' : test_ssim,
            'epoch' : epoch,
            'lr' : current_lr  # todo: no get_last_lr() attribute for GFNet
        }
        

        # log_dict.update(model.module.log_dict)
        pbar.set_postfix({'train_psnr' : f'{train_psnr:4g}',
                          'val_psnr' : f'{test_psnr:4g}',
                          'LR' : f'{current_lr:4g}'})

        torch.save({'epoch' : epoch,
                    'config' : config["model"],
                    'model' : model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'test_loss' : test_loss,
                    'test_psnr' : test_psnr,
                    'test_ssim' : test_ssim,
                    'train_loss' : train_loss,
                    'train_psnr' : train_psnr,
                    'best_psnr' : best_psnr}, f'{experiment_dir}/model_{epoch+1:03d}.pth')

        remove_previous_models = 5
        if remove_previous_models:
            fname = f'{experiment_dir}/model_{epoch-remove_previous_models:03d}.pth'
            if os.path.isfile(fname):
                os.remove(fname)


        if test_psnr > best_psnr:
            best_psnr = test_psnr
            torch.save({'model' : model.state_dict(),
                        'config' : config["model"],
                        'best_psnr' : best_psnr}, f'{experiment_dir}/best_model.pth')
        
        log_dict.update({'best_psnr' : best_psnr})
        if not config["debug"]: 
            wandb.log(log_dict)
        

        # lr scheduling
        scheduler.step()
    if not config["debug"]: 
        wandb.watch(model, criterion, log="all", log_freq=50, log_graph=True)

def sweep(args):
    wandb.init(project="first-sweep")
    try:
        args["criterion"]["eps"] = wandb.config.eps
        print("New Args: ", wandb.config)
    except:
        print("init state")
    joined = mp.spawn(main, args=(args,), nprocs=args["world_size"])
    score = np.load("result.npy")
    wandb.log({"score": score})
    
if __name__ == "__main__":

    args = Experiment().parse()
    if not 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args["gpus"]}'
    
    main(args)
