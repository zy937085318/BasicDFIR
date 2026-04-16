from .fmowdatases import Fmow
from .lrhedatasets import Lrhrdataset

def load_dataset(name,config):
    if name == "fmow":
        data_list = config["datasets"]["datalist"]
        fmowdatasets = Fmow(data_list)
        return fmowdatasets
    if name == "lrnrdatasets":
        hrdata_list = config["datasets"]["datalist"]
        lrdata_list = config["datasets"]["datalistlr"]
        dataset_type = config["train_type"]
        datasets = Lrhrdataset(hrdata_list,lrdata_list,dataset_type)
        return datasets