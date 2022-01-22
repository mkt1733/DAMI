"""
@File    : predict.py
@Time    : 2021/8/25 20:36
@Author  : Makoto
@Email   : yucheng.zhang@tum.de
@Software: PyCharm
"""

from tqdm import tqdm
from torchvision.utils import save_image
from torch.autograd import Variable
from models import *
from datasets import *

if __name__ == '__main__':

    # set parameters

    cfg = get_cfg()

    model_path = path_adapter(cfg.cat_saved_model_path)
    img_path = path_adapter(cfg.cat_image_path)
    save_to_path = path_adapter(cfg.cat_save_to)
    os.makedirs('%s/%s' % (save_to_path, cfg.cat_dataset_name), exist_ok=True)

    # set model

    generator = GeneratorUNet()
    generator.cuda()
    generator.load_state_dict(torch.load(model_path))

    # convert image

    image_filenames = [x for x in listdir(img_path) if is_image_file(x)]
    Tensor = torch.cuda.FloatTensor

    for img_name in tqdm(image_filenames):
        imgpath = img_path + '/' + img_name
        not_generated_list = []
        # cut image
        img = Image.open(imgpath)
        y = 0
        x = 0
        while y < 1281:
            while x < 1793:
                cropped = img.crop((x, y, x+256, y+256))
                tensor_cropped = transforms.ToTensor()(cropped)
                tensor_cropped = tensor_cropped.float()
                not_generated_list.append(tensor_cropped)
                x += 256
            x = 0
            y += 256
        # feed into genertor
        generated_list = []
        for img_frag in not_generated_list:
            # load image
            orig_img = img_frag.unsqueeze(0)
            orig_img = Variable(orig_img.type(Tensor))
            # into generator
            generated_img = generator(orig_img)
            # generated_img = orig_img
            generated_list.append(generated_img)
        # concatenate image in generated list
        row_list = []
        column_list = []
        count = 0
        for g_img in generated_list:
            row_list.append(g_img)
            count += 1
            if (count % 8) == 0:
                _row = torch.cat(row_list, -1)
                column_list.append(_row)
                row_list = []
                count = 0
        final_img = torch.cat(column_list, -2)
        # save final big image
        save_image(final_img, '%s/%s/%s' % (save_to_path, cfg.cat_dataset_name, img_name), normalize=False)
        # Caution: parameter |normalize| in func save_image() must be False.
        torch.cuda.empty_cache()
    print("done.")
