import torch
from torch.utils.data import DataLoader

from datasets.dataset import NPY_datasets, Test_datasets
from tensorboardX import SummaryWriter
from models import *

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

from ptflops import get_model_complexity_info

def compute_dice(pred, target, smooth=1e-6): 
    intersection = (pred * target).sum() 
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def compute_iou(pred, target, smooth=1e-6): 
    intersection = (pred * target).sum() 
    union = pred.sum() + target.sum() - intersection 
    return (intersection + smooth) / (union + smooth)

def evaluate_metrics(model, dataloader, device, config=None):
    model.eval()
    dice_total, iou_total, count = 0.0, 0.0, 0
    # sigmoid가 이미 적용된 모델들
    sigmoid_applied_models = ['egeunet', 'MHorUNet', 'malunet', 'AttU_Net', 'CMUNeXt', 'HFUNet', 'MAResUNet', 'MHA_UNet']
    with torch.no_grad():
        for data in dataloader:
            images, targets = data
            images = images.to(device).float()
            targets = targets.to(device)
            outputs = model(images)
            # AMNet의 경우 딕셔너리 출력
            if isinstance(outputs, dict):
                outputs = outputs['out']
            while isinstance(outputs, tuple):
                outputs = outputs[0]
            # sigmoid가 이미 적용된 모델인지 확인
            if config is not None and config.network in sigmoid_applied_models:
                preds = (outputs > 0.5).float()
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float()
            dice_total += compute_dice(preds, targets).item()
            iou_total += compute_iou(preds, targets).item()
            count += 1
    model.train()
    return dice_total / count, iou_total / count

def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    device = torch.device("cuda:0")
    # work_dir, log_dir, checkpoint_dir 등 기본 설정 (필요하다면 exp_idx를 포함한 폴더로 업데이트)
    for exp in range(1, 11):  # 1번부터 10번까지 반복
        print(f'#---------- Experiment {exp} ----------#')
        
        # 각 실험마다 별도의 작업 디렉토리 설정 (예: results/lbunet_isic18_월_일_시분초/exp_1)
        exp_work_dir = os.path.join(config.work_dir, f"exp_{exp}")
        os.makedirs(exp_work_dir, exist_ok=True)
        # Logger, SummaryWriter 등도 exp_work_dir 내에 생성 (생략 가능)
        log_dir = os.path.join(exp_work_dir, 'log')
        checkpoint_dir = os.path.join(exp_work_dir, 'checkpoints')
        outputs = os.path.join(exp_work_dir, 'outputs')
        for d in [log_dir, checkpoint_dir, outputs]:
            if not os.path.exists(d):
                os.makedirs(d)

        global logger
        logger = get_logger('train_' + str(exp), log_dir)
        global writer
        writer = SummaryWriter(os.path.join(exp_work_dir, 'summary'))

        log_config_info(config, logger)



        print('#----------GPU init----------#')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
        set_seed(config.seed)
        torch.cuda.empty_cache()



        print('#----------Preparing dataset----------#')
        train_dataset = NPY_datasets(config.data_path, config, train=True, exp_idx=exp)
        train_loader = DataLoader(train_dataset,
                                    batch_size=config.batch_size, 
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=config.num_workers)
        val_dataset = NPY_datasets(config.data_path, config, train=False, exp_idx=exp)
        val_loader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    drop_last=False)
        # 테스트 데이터셋 추가 (별도의 test 폴더에서 데이터를 불러옴)
        test_dataset = Test_datasets(config.data_path, config, exp_idx=exp)
        test_loader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=config.num_workers,
                                    drop_last=False)



        print('#----------Prepareing Model----------#')
        model_cfg = config.model_config
        if config.network == 'egeunet':
            model = EGEUNet(num_classes=model_cfg['num_classes'], 
                            input_channels=model_cfg['input_channels'], 
                            c_list=model_cfg['c_list']
                            )
        elif config.network == 'ucmnet':
            model = UCMNet(num_classes=model_cfg['num_classes'], 
                           in_channels=model_cfg['input_channels']
                           )
        elif config.network == 'tinyunet':
            model = TinyUNet(num_classes=model_cfg['num_classes'], 
                            in_channels=model_cfg['input_channels']
                            )
        elif config.network == 'unet':
            model = UNet(in_channels=model_cfg['in_channels'], 
                        out_channels=model_cfg['out_channels'], 
                        features=model_cfg.get('features', [64, 128, 256, 512])
                        )
        elif config.network == 'MHorUNet':
            model = MHorUNet(num_classes=model_cfg['num_classes'], 
                    input_channels=model_cfg['input_channels'], 
                    c_list=model_cfg['c_list'], 
                    split_att=model_cfg['split_att'], 
                    bridge=model_cfg['bridge'],
                    drop_path_rate=model_cfg['drop_path_rate']
                    )
        elif config.network == 'propose':
            model = propose(num_classes=model_cfg['num_classes'],
                            input_channels=model_cfg['in_channels'],
                            c_list=model_cfg['c_list'],
                            )
        elif config.network == 'UltraLight_VM_UNet':
            model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'],
                            input_channels=model_cfg['input_channels'],
                            c_list=model_cfg['c_list'],
                            split_att=model_cfg['split_att'],
                            bridge=model_cfg['bridge']
                            )
        elif config.network == 'amnet':
            model = AMNet(num_classes=model_cfg['num_classes'],
                          input_channels=model_cfg['input_channels'],
                          base_c=model_cfg['base_c'],
                          bilinear=model_cfg['bilinear']
                          )
        elif config.network == 'malunet':
            model = MALUNet(num_classes=model_cfg['num_classes'],
                            input_channels=model_cfg['input_channels'],
                            c_list=model_cfg['c_list'],
                            split_att=model_cfg['split_att'],
                            bridge=model_cfg['bridge']
                            )
        elif config.network == 'AttU_Net':
            model = AttU_Net(num_classes=model_cfg['num_classes'],
                             in_channel=model_cfg['in_channel'],
                             channel_list=model_cfg['channel_list'],
                             checkpoint=model_cfg['checkpoint'],
                             convTranspose=model_cfg['convTranspose']
                             )
        elif config.network == 'CMUNeXt':
            model = CMUNeXt(num_classes=model_cfg['num_classes'],
                            input_channel=model_cfg['input_channel'],
                            dims=model_cfg['dims'],
                            depths=model_cfg['depths'],
                            kernels=model_cfg['kernels']
                            )
        elif config.network == 'HFUNet':
            model = HFUNet(num_classes=model_cfg['num_classes'],
                           input_channels=model_cfg['input_channels'],
                           c_list=model_cfg['c_list'],
                           bridge=model_cfg['bridge']
                           )
        elif config.network == 'MAResUNet':
            model = MAResUNet(num_classes=model_cfg['num_classes'],
                              num_channels=model_cfg['num_channels']
                              )
        elif config.network == 'MHA_UNet':
            model = MHA_UNet(num_classes=model_cfg['num_classes'],
                             input_channels=model_cfg['input_channels'],
                             c_list=model_cfg['c_list'],
                             split_att=model_cfg['split_att'],
                             bridge=model_cfg['bridge']
                             )
        else: raise Exception('network in not right!')
        model = model.cuda()

        # 추가: 모델의 FLOPs와 파라미터 수 측정 (입력 크기: (채널, 256, 256))
        if config.network in ['egeunet', 'ucmnet', 'tinyunet', 'malunet', 'MHorUNet', 'UltraLight_VM_UNet', 'amnet', 'HFUNet', 'MHA_UNet']:
            dummy_input = (model_cfg['input_channels'], 256, 256)
        elif config.network in ['CMUNeXt']:
            dummy_input = (model_cfg['input_channel'], 256, 256)
        elif config.network in ['AttU_Net']:
            dummy_input = (model_cfg['in_channel'], 256, 256)
        elif config.network in ['MAResUNet']:
            dummy_input = (model_cfg['num_channels'], 256, 256)
        else:
            dummy_input = (model_cfg['in_channels'], 256, 256)
        flops, params = get_model_complexity_info(model, dummy_input, as_strings=True, print_per_layer_stat=False)
        logger.info("Model FLOPs: %s, Params: %s", flops, params)

        print('#----------Prepareing loss, opt, sch and amp----------#')
        criterion = config.criterion
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)


        print('#----------Set other params----------#')
        min_value = 999
        start_epoch = 1
        min_epoch = 1


        step = 0
        print('#----------Training----------#')
        for epoch in range(start_epoch, config.epochs + 1):

            torch.cuda.empty_cache()

            step = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                step,
                logger,
                config,
                writer
            )

            value = val_one_epoch(
                    val_loader,
                    model,
                    config.criterion,
                    epoch,
                    logger,
                    config,
                    writer
                )

            val_dice, val_iou = evaluate_metrics(model, val_loader, device, config)
            logger.info("Epoch {}: Validation Dice: {:.4f}, IoU: {:.4f}".format(epoch, val_dice, val_iou))
            writer.add_scalar("Val/Dice", val_dice, epoch)
            writer.add_scalar("Val/IoU", val_iou, epoch)

            if value < min_value:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_value = value
                min_epoch = epoch


        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
            model.load_state_dict(best_weight)
            # exp_work_dir 기준의 outputs 폴더 경로 생성
            out_path = os.path.join(exp_work_dir, 'outputs/')
            test_one_epoch(
                    test_loader,
                    model,
                    config.criterion,
                    logger,
                    config,
                    path = out_path,
                    writer=writer
                )
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'best-epoch{min_epoch}.pth')
            )    


if __name__ == '__main__':
    config = setting_config
    main(config)