import os


def load_dataset(args):
    import tools
    img_0_1 = tools.import_from_path(os.path.dirname(__file__) + '/img_0_1.py')
    dataset = img_0_1.load_dataset(args)
    return dataset * 255
