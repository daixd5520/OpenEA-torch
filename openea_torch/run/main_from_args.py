import argparse
import sys
import time

from ..src.openea_torch.modules.args.args_hander import check_args, load_args
from ..src.openea_torch.modules.load.kgs import read_kgs_from_folder
from ..src.openea_torch.models.trans.transe import TransE  # Assuming you have a PyTorch version of TransE
from ..src.openea_torch.models.trans.transh import TransH  # Assuming you have a PyTorch version of TransH
from ..src.openea_torch.approaches import BootEA
from ..src.openea_torch.models.basic_model import BasicModel  # Assuming you have a PyTorch version of BasicModel

class ModelFamily(object):
    BasicModel = BasicModel
    TransE = TransE
    TransH = TransH
    BootEA = BootEA

def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    t = time.time()
    args = load_args(sys.argv[1])
    args.training_data = args.training_data + sys.argv[2] + '/'
    args.dataset_division = sys.argv[3]
    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    # if args.embedding_module == "RSN4EA":
    #     remove_unlinked = True
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    
    # Assuming that the `run` method is implemented correctly in the PyTorch version of the model.
    model.run()
    model.test()
    model.save()
    
    print("Total run time = {:.3f} s.".format(time.time() - t))