if __name__ == "__main__":
    from torch.optim import Adam

    from engines.SegTrainer import SegTrainer
    from dataset.data_loaders import getDataloaders
    from dataset.data_prepare import generate_sdms, get_tts
    from utils.util import initModel
    from configs import CONF

    def main(gpu_id, imIDs):
        print(f'[INFO] MODEL: {CONF.MODEL}, DATASET: {CONF.DSET}, LOSS: {CONF.LOSS}') 
        loaders = getDataloaders(CONF, gpu_id, imIDs)
        model = initModel(gpu_id)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONF.INIT_LR, weight_decay=CONF.L2_DECAY)
        SegTrainer(gpu_id, model, optimizer, loaders).train()

    imIDs = get_tts()
    generate_sdms(imIDs)
    print(f'[INFO] Running in non-distributed mode on {CONF.DEFAULT_DEVICE}')
    main(CONF.DEFAULT_DEVICE, imIDs)
  

    