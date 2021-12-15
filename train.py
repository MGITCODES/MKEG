import time
import argparse
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from dataset import *
from loss import *
import visual as Visualmod
import text_decoder as Gmod
import diseaseClassifier as classification
import os
torch.backends.cudnn.enabled = False
from tqdm import tqdm


class DebuggerBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_train_loss = 10000000000
        self.params = None
        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.vocab = self._init_vocab()
        self.model_state_dict = self._load_mode_state_dict()
        self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform)
        self.extractor = self._init_visual_extractor()
        self.gcn = self._init_gcn()
        self.diseaseClassifier = self._init_diseaseClassifier()
        self.word_model = self._init_word_model()
        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.writer.write("{}\n".format(self.args))

    def train(self):
        for epoch_id in range(self.start_epoch, self.args.epochs):
            train_class_loss, train_word_loss, train_loss = self._epoch_train()
            val_class_loss, val_word_loss, val_loss = self._epoch_val()

            if self.args.mode == 'train':
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step(val_loss)
            self.writer.write(
                "[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}\n".format(self._get_now(),
                                                                               epoch_id,
                                                                               train_loss,
                                                                               val_loss,
                                                                               self.optimizer.param_groups[0]['lr']))
            print("[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}\n".format(self._get_now(),
                                                                                 epoch_id,
                                                                                 train_loss,
                                                                                 val_loss,
                                                                                 self.optimizer.param_groups[0]['lr']))
            print("tag_loss:{},word_loss:{}".format(train_class_loss, train_word_loss))
            self._save_model(epoch_id,
                             val_loss,
                             val_class_loss,
                             val_word_loss,
                             train_loss)

    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize,self.args.resize)),
            # transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # model_dir = os.path.join(model_dir, self._get_now())
        model_dir = model_dir + str(self._get_now().replace(':', '')) + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        self.writer.write("Vocab Size:{}\n".format(len(vocab)))

        return vocab


    def _load_mode_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch']
            self.writer.write("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            self.writer.write("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):

        model = Visualmod.VisualFeatureExtractor()

        try:
            model_state = torch.load(self.args.load_visual_model_path)
            model.load_state_dict(model_state['extractor'])
            self.writer.write("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()

        return model

    def _init_gcn(self):
        model = Gmod.GCN_Encoder1(n_feature=self.args.gcnin, n_hidden=self.args.gcnout)

        try:
            model_state = torch.load(self.args.load_gcn_model_path)
            model.load_state_dict(model_state['gcn'])
            self.writer.write("[Load gcn Succeed!]\n")
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()

        return model

    def _init_diseaseClassifier(self):
        model = classification.diseaseClassifier(n_feature=self.args.sementic_features_dim, n_output=self.args.classes)

        try:
            model_state = torch.load(self.args.load_mlc_model_path)
            model.load_state_dict(model_state['diseaseClassifier'])

        except Exception as err:
            self.writer.write("[Load diseaseClassifier Failed {}!]\n".format(err))

        if not self.args.mlc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model


    def _init_word_model(self):
        raise NotImplementedError

    def _init_data_loader(self, file_list, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=True)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.BCELoss()
        # nn.MSELoss()

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self,
                    epoch_id,
                    val_loss,
                    val_class_loss,
                    val_word_loss,
                    train_loss):
        def save_whole_model(_filename):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'class': self.diseaseClassifier.state_dict(),
                        'gcn': self.gcn.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        def save_part_model(_filename, value):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({"model": value},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        if val_loss < self.min_val_loss:
            file_name = "val_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_val_loss = val_loss

        if train_loss < self.min_train_loss:
            file_name = "train_best_loss.pth"+str(epoch_id)+".tar"
            save_whole_model(file_name)
            self.min_train_loss = train_loss


class TransformerDebugger(DebuggerBase):
        def _init_(self, args):
            DebuggerBase.__init__(self, args)
            self.args = args

        def _epoch_train(self):

            tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
            edge = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 9, 10, 10, 11,
                                  11, 12, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,
                                  17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20,
                                  20, 20, 20],
                                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 8, 7, 10, 11,
                                  9, 11, 9, 10, 13, 12, 15, 16, 17, 18, 19, 20, 14, 16, 17, 18, 19, 20, 14, 15, 17, 18,
                                  19, 20, 14, 15, 16, 18, 19, 20, 14, 15, 16, 17, 19, 20, 14, 15, 16, 17, 18, 20, 14,
                                  15, 16, 17, 18, 19]], dtype=torch.long)
            edge = edge.cuda()
            self.extractor.train()
            self.diseaseClassifier.train()
            self.gcn.train()
            self.word_model.train()

            for i, (img_front, img_later, _, label, captions, prob) in enumerate(self.train_data_loader):
                batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
                img_front = self._to_var(img_front)
                img_later = self._to_var(img_later)

                # visual_features, avg_features = self.extractor.forward(images)
                front = self.extractor.forward(img_front)
                later = self.extractor.forward(img_later)

                front = front.view(front.shape[0], front.shape[1], -1)
                later = later.view(later.shape[0], later.shape[1], -1)

                f_global = torch.sum(front,dim=1)/20
                l_global = torch.sum(later,dim=1)/20

                f_global = f_global.unsqueeze(1)
                l_global = l_global.unsqueeze(1)

                front = torch.cat([f_global, front], dim=1)
                later = torch.cat([l_global, later], dim=1)

                front = self.gcn.forward(front, edge)
                later = self.gcn.forward(later, edge)

                prev_hidden_states = torch.cat([front[:, 0, :].unsqueeze(1), later[:, 0, :].unsqueeze(1)], dim=2)
                vis_feature = torch.cat([front[:, 1:, :], later[:, 1:, :]], dim=2)

                classes = self.diseaseClassifier.forward(vis_feature).squeeze(2)



                context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
                prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)


                batch_loss = + self.args.lambda_word * batch_word_loss

                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm(self.word_model.parameters(), self.args.clip)
                self.optimizer.step()

                # tag_loss += self.args.lambda_tag * batch_tag_loss.data
                word_loss += self.args.lambda_word * batch_word_loss.data
                loss += batch_loss.data

            return tag_loss, stop_loss, word_loss, loss

        def _epoch_val(self):
            class_loss, word_loss, loss = 0, 0, 0
            return class_loss, word_loss, loss

        def evaluate_loss(model, dataloader, loss_fn, text_field):

            # Validation loss
            model.eval()
            running_loss = .0
            with tqdm(desc='Epoch %d - validation', unit='it', total=len(dataloader)) as pbar:
                with torch.no_grad():
                    for it, (detections, captions) in enumerate(dataloader):
                        detections, captions = detections, captions
                        out = model(detections, captions)
                        captions = captions[:, 1:].contiguous()
                        out = out[:, :-1].contiguous()
                        loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                        this_loss = loss.item()
                        running_loss += this_loss

                        pbar.set_postfix(loss=running_loss / (it + 1))
                        pbar.update()

            val_loss = running_loss / len(dataloader)
            return val_loss


        def train_xe(model, dataloader, optim, text_field, loss_fn):
            # Training with cross-entropy
            model.train()

            print('lr = ', optim.state_dict()['param_groups'][0]['lr'])

            running_loss = .0
            with tqdm(desc='Epoch %d - train', unit='it', total=len(dataloader)) as pbar:
                for it, (detections, captions) in enumerate(dataloader):
                    detections, captions = detections, captions
                    out = model(detections, captions)
                    optim.zero_grad()
                    captions_gt = captions[:, 1:].contiguous()
                    out = out[:, :-1].contiguous()
                    loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                    loss.backward()

                    optim.step()
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()
                    # scheduler.step()

            loss = running_loss / len(dataloader)
            return loss



if __name__ == '__main__':
    import warnings

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='D:/Medical-Report-Generation-master/data/new_data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='D:/Medical-Report-Generation-master/data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='D:/Medical-Report-Generation-master/data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='D:/gcn/train.txt',
                        help='the train array')

    # transforms argument
    parser.add_argument('--resize', type=int, default=512,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='size for randomly cropping images')
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./report_v4_models/',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str, default='D:/gcn/report_v4_models/v420210119-0355/train_best_loss.pth27.tar',
                        help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='v4',
                        help='The name of saved model')

    """
    Model Argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='not using pretrained model when training')
    parser.add_argument('--load_visual_model_path', type=str,
                        default='D:/gcn/report_v4_models/v420210119-0355/train_best_loss.pth27.tar')
    parser.add_argument('--visual_trained', action='store_true', default=True,
                        help='Whether train visual extractor or not')

    # 图结构
    parser.add_argument('--gcnin', type=int, default=256)
    parser.add_argument('--gcnout', type=int, default=256)
    parser.add_argument('--load_gcn_model_path', type=str,
                        default='D:/gcn/report_v4_models/v420210119-0355/train_best_loss.pth27.tar')


    # MLC
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--load_mlc_model_path', type=str,
                        default='D:/gcn/report_v4_models/v420210119-0355/train_best_loss.pth27.tar')
    parser.add_argument('--mlc_trained', action='store_true', default=True)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v4')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_co_model_path', type=str, default='')
    parser.add_argument('--co_trained', action='store_true', default=True)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--visual_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--load_sentence_model_path', type=str,
                        default='D:/gcn/report_v4_models/v420210119-0355/train_best_loss.pth27.tar')
    parser.add_argument('--sentence_trained', action='store_true', default=True)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)
    parser.add_argument('--load_word_model_path', type=str,
                        default='D:/gcn/report_v4_models/v420210119-0355/train_best_loss.pth27.tar')
    parser.add_argument('--word_trained', action='store_true', default=True)

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    debugger = TransformerDebugger(args)
    debugger.train()
