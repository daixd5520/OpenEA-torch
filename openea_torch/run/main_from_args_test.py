import argparse
import sys
import time

from openea_torch.modules.args.args_hander import check_args, load_args
from openea_torch.modules.load.kgs import read_kgs_from_folder
from openea_torch.models.trans import TransE
from openea_torch.models.trans import TransH
from openea_torch.approaches import BootEA
from openea_torch.models.basic_model import BasicModel


class ModelFamily(object):
    BasicModel = BasicModel

    TransE = TransE
    TransH = TransH

def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    t = time.time()
    args = load_args(sys.argv[1])
    args.training_data = args.training_data + sys.argv[2] + '/'
    args.dataset_division = sys.argv[3]
    print(args)
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.retest()
    print("Total run time = {:.3f} s.".format(time.time() - t))

