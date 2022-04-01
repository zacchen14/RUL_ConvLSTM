import time
from convlstm import EncoderDecoder
import torch
import numpy as np
import torch.utils.data as Data
from torch.optim import Adam
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse
import logging
from common.loss import RMSELoss
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_logger(filename, verbosity=1, name=None):
    """

    Args:
        filename:
        verbosity:
        name:

    Returns:

    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train(model, train_loader, test_loader, logger, n_epochs, criterion, opt):
    logger.info(args)
    logger.info('Start training!')

    scheduler = StepLR(opt, step_size=50, gamma=0.1)
    model.train()
    for epoch in range(n_epochs):
        model.zero_grad()
        epoch_loss = 0.
        step = None
        for step, (batch_x, batch_y) in enumerate(tqdm(train_loader, mininterval=0.5)):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model.forward(batch_x)
            loss = criterion(output, batch_y.unsqueeze(dim=-1))

            loss.backward()
            opt.step()
            model.zero_grad()

            epoch_loss += loss

        epoch_loss /= step
        eval_loss = evaluation(model, test_loader, criterion)
        scheduler.step()

        print('\r epoch: %d / %d, err_rul: %f, eval_rul: %f'
              % (epoch, n_epochs, epoch_loss.cpu().detach().numpy(), eval_loss))
        logger.info('Epoch:[{}/{}]\t train_loss={:.5f} validation_loss={:.5f}'.format
                    (epoch, n_epochs, epoch_loss, eval_loss))

    logger.info('Finish training!')


def evaluation(model, loader, criterion):
    model.eval()
    loss = 0.
    step = None
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model.forward(batch_x)
        output = torch.where(output <= 125, output, 125 * torch.ones(output.size()).to(device))
        batch_loss = criterion(output, batch_y.unsqueeze(dim=-1))

        loss += batch_loss

    loss /= step

    return loss


def test(model, input: torch.Tensor, label):
    """

    Args:
        model:
        input: (batch, T, input_channel, height, width)
        label:
        criterion:

    Returns:

    """
    criterion = nn.MSELoss()
    input = input.to(device)
    output = model.forward(input).squeeze(-1).detach().cpu()
    # rul_target = torch.tensor(np.loadtxt("data/CMAPSSData/RUL_FD001.txt"))
    loss = criterion(output, label)
    return loss


def main(args: argparse.Namespace):
    criterion = torch.nn.MSELoss()
    print('FD004')
    # Create dataloader
    x_train = torch.tensor(
        np.load(file="data/CMAPSSData/"+str(args.twlen)+"_"+str(args.T)+"/train_x_FD004.npy"), dtype=torch.float32)
    y_train = torch.tensor(
        np.load(file="data/CMAPSSData/"+str(args.twlen)+"_"+str(args.T)+"/train_y_FD004.npy"), dtype=torch.float32)
    x_test = torch.tensor(
        np.load(file="data/CMAPSSData/"+str(args.twlen)+"_"+str(args.T)+"/test_x_FD004.npy"), dtype=torch.float32)
    y_test = torch.tensor(
        np.load(file="data/CMAPSSData/"+str(args.twlen)+"_"+str(args.T)+"/test_y_FD004.npy"), dtype=torch.float32)
    train_ds: Data.TensorDataset = Data.TensorDataset(x_train, y_train)
    test_ds: Data.TensorDataset = Data.TensorDataset(x_test, y_test)
    train_loader = Data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = Data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # set up the model and optimizer
    convLSTM_model = EncoderDecoder(input_dim=args.in_channel, conv_dim=args.conv_dim, kernel_size=args.kernel_size,
                                    num_layers=args.num_layers).to(device)
    Adam_optimizer = Adam(convLSTM_model.parameters(), lr=args.lr)

    logger = get_logger('./log/' + str(time.time()) + '.log')
    train(convLSTM_model, train_loader, test_loader, logger, n_epochs=args.epochs, criterion=criterion, opt=Adam_optimizer)

    case_x = torch.tensor(
        np.load(file="data/CMAPSSData/"+str(args.twlen)+"_"+str(args.T)+"/case_x_FD004.npy"), dtype=torch.float32)
    case_y = torch.tensor(
        np.load(file="data/CMAPSSData/"+str(args.twlen)+"_"+str(args.T)+"/case_y_FD004.npy"), dtype=torch.float32)
    test_loss = test(convLSTM_model, case_x, case_y)
    logger.info('Loss in test_dataset: {:.3f}'.format(test_loss))

    if args.save:
        torch.save(convLSTM_model, 'model/'+str(args.twlen)+'_'+str(args.T)+'/'+str(args.epochs)+'.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allan Analysis for RUL prediction')

    # training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--save', action='store_false',
                        help='Whether save the model(Default: True)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # data format parameters
    parser.add_argument('--twlen', type=int, default=16,
                        help='data processing parameter')
    parser.add_argument('--T', type=int, default=8,
                        help='data processing parameter')
    # model parameters
    parser.add_argument('--in_channel', type=int, default=1,
                        help='input_channel of model, 3 for RGB image, and 1 for time series')
    parser.add_argument('--conv_dim', type=int, default=1,
                        help='the number of filters')
    parser.add_argument('--kernel_size', type=tuple, default=(5, 5),
                        help='kernel_size')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='num_layers')

    args = parser.parse_args()
    main(args)
