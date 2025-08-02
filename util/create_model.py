from model.unet import UNet
def create_model(opt):
    if opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    else:
        model = UNet()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model