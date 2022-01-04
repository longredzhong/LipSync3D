from torch import optim
from torch.optim import optimizer

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DMeshDataset
from model import Lipsync3DMesh
from loss import L2Loss
from audiodvp_utils.visualizer import Visualizer
import time
import os

import torch.nn as nn

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device

    dataset = Lipsync3DMeshDataset(opt)
    train_dataloader = DataLoader(
        dataset,
        batch_size = opt.batch_size,
        shuffle = not opt.serial_batches, # default not shuffle
        num_workers = opt.num_workers,
        drop_last = True
    )

    visualizer = Visualizer(opt)
    model = Lipsync3DMesh().to(device)

    #TODO : Define Loss function------
    criterionGeo = None
    #---------------------------------

    if opt.load_model:
        if os.path.exists(os.path.join(opt.tgt_dir, opt.model_name)):
            state_dict = torch.load(os.path.join(opt.tgt_dir, opt.model_name))
            audioEncoder_state = {}
            geometryDecoder_state = {}

            for key, value in state_dict.items():
                if 'AudioEncoder' in key:
                    audioEncoder_state[key.replace('AudioEncoder.', '')] = value
                if 'GeometryDecoder' in key:
                    geometryDecoder_state[key.replace('GeometryDecoder.', '')] = value

            model.AudioEncoder.load_state_dict(audioEncoder_state)
            model.GeometryDecoder.load_state_dict(geometryDecoder_state)
           
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    os.makedirs(os.path.join(opt.tgt_dir, 'mesh_checkpoint'), exist_ok=True)

    # model = nn.DataParallel(model)

    total_iters = 0

    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # TODO : Implement training process -------
            geoLoss = None
            # -----------------------------------------

            if total_iters % opt.print_freq == 0:
                losses = {'geoLoss' : geoLoss}

                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

        if epoch % opt.checkpoint_interval == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(opt.tgt_dir, 'mesh_checkpoint', 'checkpoint_{}.pth'.format(epoch)))
            print("Checkpoint saved")

    torch.save(model.state_dict(), os.path.join(opt.tgt_dir, 'mesh.pth'))
