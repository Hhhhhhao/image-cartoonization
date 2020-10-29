import os
working_dir = os.path.dirname(__file__)
import argparse
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
from data_loaders import
from models import Generator
from utils.misc import read_json


def get_config(manual=None):
    parser = argparse.ArgumentParser('Image Cartoon')
    # basic options
    parser.add_argument('--checkpoint-path', default='', help='checkpoint path')
    return parser.parse_args(manual)


def main():
    config = get_config()

    # find config.json in checkpoint folder
    checkpoint_path = os.path.join(working_dir, config.checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    exp_dir = os.path.dirname(checkpoint_dir)
    result_dir = os.path.join(exp_dir, 'results')

    # make device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load config
    config = read_json(os.path.join(exp_dir, 'config.json'))
    config = edict(config)

    # set parameters of config
    config.num_workers = 0

    # build dataloader
    data_loader =

    # build model
    model = Generator(config.image_size, config.down_size, config.num_res, config.skip_conn)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['gen_state_dict'])
    model.to(device)
    model.eval()

    # start evaluation
    print("start evaluation")
    results = []
    with torch.no_grad():
        for batch_idx, (uttr, uttr_len) in tqdm(enumerate(data_loader), total=len(data_loader)):
            uttr = uttr.to(device)
            uttr_len = uttr_len.to(device)

            out, out_lens = model(uttr, uttr_len)

            # decode
            pred_phon, _, _, pred_phon_lens = beam_decoder.decode(out.transpose(0, 1), out_lens)
            best_phon = pred_phon[:, 0].cpu().numpy()
            best_phon_lens = pred_phon_lens[:, 0].cpu().numpy()

            for i in range(best_phon.shape[0]):
                # convert to string
                hypo = best_phon[i, :best_phon_lens[i]]
                phoneme = ''.join(PHONEME_MAP[idx] for idx in hypo)
                results.append(phoneme)

    with open(os.path.join(result_dir, 'test_result.csv'), "w") as f:
        f.write('id,label\n')
        for i, phoneme in enumerate(results):
            f.write('{:d},{:s}\n'.format(i, phoneme))


if __name__ == '__main__':
    main()