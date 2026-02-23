import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        # 모델 출력 처리 (모든 모델은 raw logits 출력)
        model_output = model(images)

        if isinstance(model_output, dict):  # AMNet 등 딕셔너리 출력
            out = model_output['out']
            gt_pre = (out, out, out, out, out)
        elif isinstance(model_output, tuple):  # Deep supervision 모델
            gt_pre, out = model_output
        else:  # 단일 출력
            out = model_output
            gt_pre = (out, out, out, out, out)

        # 모든 모델에 sigmoid 적용
        gt_pre = tuple(torch.sigmoid(x) for x in gt_pre)
        out = torch.sigmoid(out)

        loss = criterion(gt_pre, out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    writer):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in test_loader:
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # 모델 출력 처리 (모든 모델은 raw logits 출력)
            model_output = model(img)

            if isinstance(model_output, dict):  # AMNet 등 딕셔너리 출력
                out = model_output['out']
                gt_pre = (out, out, out, out, out)
            elif isinstance(model_output, tuple):  # Deep supervision 모델
                gt_pre, out = model_output
            else:  # 단일 출력
                out = model_output
                gt_pre = (out, out, out, out, out)

            # 모든 모델에 sigmoid 적용
            gt_pre = tuple(torch.sigmoid(x) for x in gt_pre)
            out = torch.sigmoid(out)

            # 수치 안정성: [0, 1] 범위로 clamp (BCE 에러 방지)
            if isinstance(gt_pre, tuple):
                gt_pre = tuple(torch.clamp(x, 1e-7, 1 - 1e-7) for x in gt_pre)
            else:
                gt_pre = torch.clamp(gt_pre, 1e-7, 1 - 1e-7)
            out = torch.clamp(out, 1e-7, 1 - 1e-7)

            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            # BETA TEST: 1 batch만
            if hasattr(config, 'BETA_TEST') and config.BETA_TEST:
                break

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        logger.info(log_info)
        # SummaryWriter에 기록
        writer.add_scalar("Val/Loss", np.mean(loss_list), epoch)
        writer.add_scalar("Val/MIoU", miou, epoch)
        writer.add_scalar("Val/Dice", f1_or_dsc, epoch)
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        writer.add_scalar("Val/Sensitivity", sensitivity, epoch)
        writer.add_scalar("Val/Specificity", specificity, epoch)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        logger.info(log_info)
        writer.add_scalar("Val/Loss", np.mean(loss_list), epoch)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    path,
                    writer,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # 모델 출력 처리 (모든 모델은 raw logits 출력)
            model_output = model(img)

            if isinstance(model_output, dict):  # AMNet 등 딕셔너리 출력
                out = model_output['out']
                gt_pre = (out, out, out, out, out)
            elif isinstance(model_output, tuple):  # Deep supervision 모델
                gt_pre, out = model_output
            else:  # 단일 출력
                out = model_output
                gt_pre = (out, out, out, out, out)

            # 모든 모델에 sigmoid 적용
            gt_pre = tuple(torch.sigmoid(x) for x in gt_pre)
            out = torch.sigmoid(out)

            # 수치 안정성: [0, 1] 범위로 clamp (BCE 에러 방지)
            if isinstance(gt_pre, tuple):
                gt_pre = tuple(torch.clamp(x, 1e-7, 1 - 1e-7) for x in gt_pre)
            else:
                gt_pre = torch.clamp(gt_pre, 1e-7, 1 - 1e-7)
            out = torch.clamp(out, 1e-7, 1 - 1e-7)

            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, path, config.datasets, config.threshold, test_data_name=test_data_name)
                # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            logger.info(f'test_datasets_name: {test_data_name}')
        log_info = f'Test: Loss {np.mean(loss_list):.3f} | IoU {miou:.3f} | Dice {f1_or_dsc:.3f} | Acc {accuracy:.3f} | Sens {sensitivity:.3f} | Spec {specificity:.3f}'
        logger.info(log_info)  # 로그 파일에만 기록 (print 제거 - ex_utils.py에서 출력)
        # SummaryWriter에 기록
        writer.add_scalar("Test/Loss", np.mean(loss_list), 0)
        writer.add_scalar("Test/MIoU", miou, 0)
        writer.add_scalar("Test/Dice", f1_or_dsc, 0)
        writer.add_scalar("Test/Accuracy", accuracy, 0)
        writer.add_scalar("Test/Sensitivity", sensitivity, 0)
        writer.add_scalar("Test/Specificity", specificity, 0)

    return np.mean(loss_list)
