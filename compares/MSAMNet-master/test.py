# Basic module
from tqdm             import tqdm
from parse_args_test import  parse_args
import scipy.io as scio
import cv2
import json
from matplotlib.patches import Circle


# Torch and visulization
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.dataloader import *

# Model
from model.DNANet import  Res_CBAM_block, DNANet
from model.CSAUNet import CSAUNet
from model.DnTNet import DnTNet
from model.MSAMNet import MSAMNet, AAFE, CBAM, Connection, SE, CoordAtt

fusionblock = {'CBAM':CBAM,
             'AAFE':AAFE,
             'CA':CoordAtt,
             'SE':SE,
             'None':Connection
            }

def cal_centroid_coord(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    return num_labels - 1, centroids[1:] - 0.5


def load_split_ids(split_path):
    img_ids = []
    with open(split_path, "r") as f:
        line = f.readline()
        while line:
            sample_id = line.split("\n")[0]
            if sample_id:
                img_ids.append(sample_id)
            line = f.readline()
    return img_ids

def save_pred_txt(centroids, output_path, val_img_ids, num):
    filename = val_img_ids[num].split('/')
    txt_path = output_path + '/' + '_'.join(filename) + '.txt'
    os.makedirs(os.path.split(txt_path)[0], exist_ok=True)
    with open(txt_path, 'w') as file:
        for center in centroids:
            x_center, y_center = center
            file.write('0 %f %f %f %f\n' % (x_center/512, y_center/512, 20/512, 20/512))
            
def visual_Pred_GT(centroids_pd, centroids_gt, dataset_dir, output_path, val_img_ids, num):
    input_img_path = dataset_dir+'/'+'test'+'/images/'+val_img_ids[num]+'.tif'
    img = np.array(Image.open(input_img_path))
    img = img.clip(img.mean() - img.std(), img.mean() + 3 * img.std())
    img = img / img.max() * 255
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

    for i, center in enumerate(centroids_pd):
        x_center, y_center = center
        plt.gca().add_patch(Circle((x_center, y_center), radius=10, fill=False, edgecolor='red', linewidth=2))
    for i, center in enumerate(centroids_gt):
        x_center, y_center = center
        plt.gca().add_patch(Circle((x_center, y_center), radius=10, fill=False, edgecolor='green', linewidth=2))

    img_path = output_path + '/' + val_img_ids[num]
    os.makedirs(os.path.split(img_path)[0], exist_ok=True)
    plt.savefig(img_path)
    plt.cla()
    plt.close("all")

visual = False
class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args  = args
        self.mIoU  = mIoU(1)
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

        # Preprocess and load data from any MSOD-style dataset directory.
        dataset_dir = args.root + '/' + args.dataset
        test_txt = dataset_dir + '/' + 'test.txt'
        if not os.path.exists(test_txt):
            raise FileNotFoundError('Missing test split file: %s' % test_txt)
        val_img_ids = load_split_ids(test_txt)
        testset = MSODataset(dataset_dir, mode='test', img_ids=val_img_ids, input_size=args.input_size, num_frame=args.T_frame, suffix=args.suffix)
        
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1, input_channels=args.T_frame, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model   == 'CSAUNet':
            model       = CSAUNet(input_channels=args.T_frame)
        elif args.model   == 'DnTNet':
            model       = DnTNet(seq_length=args.T_frame)
        elif args.model   == 'MSAMNet':
            model       = MSAMNet(frame_length=args.T_frame, fusionBlock=fusionblock[args.fusionblock])
        
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        self.Floss = FocalLoss(gamma=args.loss_gamma, alpha=args.loss_alpha)

        # Load trained model
        checkpoint        = torch.load('result/' + args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])
        model_name = args.model_dir.split('/')[-1]
        epoch = checkpoint['epoch']
        print('Load epoch %d ' % (epoch) + model_name)
        if visual:
            target_image_path = dataset_dir + '/result/' + args.st_model + '_' + model_name.split('_')[0] + '_visul_result'
            if os.path.exists(target_image_path):
                shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
            os.mkdir(target_image_path)
        
        target_pred_path = dataset_dir + '/result/' +  args.st_model + '_' + model_name.split('_')[0]
        label_txt_dir  = dataset_dir + '/test/' + 'labels/'


        # Test
        self.model.eval()
        self.mIoU.reset()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        tp_num = 0
        fp_num = 0
        pred_num = 0
        gt_num = 0
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                batchsize, _, _, _ = labels.size()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                        loss += self.Floss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels) + self.Floss(pred, labels)
                losses.    update(loss.item(), pred.size(0))
                self.mIoU. update(pred, labels)

                for batch_i in range(batchsize): 
                    predsss = np.array((pred[batch_i,...] > 0).cpu()).astype('int64') * 255
                    predsss = np.uint8(predsss)
                    img = predsss.reshape(512, 512)
                    num_preds, pred_coords = cal_centroid_coord(img)

                    # targets = np.array((labels[batch_i,...] > 0).cpu()).astype('int64') * 255
                    # targets = np.uint8(targets)
                    # img = targets.reshape(512, 512)
                    # num_targets, target_coords = cal_centroid_coord(img)
                    
                    txt_path = label_txt_dir + val_img_ids[num] + '.txt'
                    num_targets = 0
                    target_coords = []
                    with open(txt_path, 'r') as f: 
                        for line in f:
                            parts = line.strip().split()
                            class_id, x, y, w, h = map(float, parts)
                            num_targets += 1
                            target_coords.append([x*512,y*512])
                    target_coords = np.array(target_coords)
                    pred_num += num_preds
                    gt_num += num_targets
                    gt_matched = np.zeros(num_targets)
                    if num_targets > 0:
                        for pred_i in range(num_preds):
                            distances = [np.linalg.norm(pred_coords[pred_i]- target_coords[t_i]) for t_i in range(num_targets)]
                            min_dis = min(distances)
                            if min_dis <= 3:
                                min_idx = distances.index(min_dis)
                                if not gt_matched[min_idx]:
                                    tp_num += 1
                                    gt_matched[min_idx] = 1
                                else:
                                    fp_num += 1
                            else:
                                fp_num += 1
                    else:
                        fp_num += num_preds
                    if visual:
                        visual_Pred_GT(pred_coords, target_coords, dataset_dir, target_image_path, val_img_ids, num)
                    save_pred_txt(pred_coords, target_pred_path, val_img_ids, num)
                    num += 1
                self.mIoU. update(pred, labels)

            pix_recall, pix_precision, mean_IOU = self.mIoU.get()
            recall      = tp_num / (gt_num  + 0.001)
            precision   = tp_num / (pred_num + 0.001)
            f1 = 2 * precision * recall / (precision + recall + 0.001)
            print('test_loss, %.4f' % (losses.avg))
            print('mean_IOU: %f pix_recall: %f pix_precision: %f'%(mean_IOU, pix_recall, pix_precision))
            print('f1: %f recall: %f precision: %f'%(f1, recall, precision))


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





