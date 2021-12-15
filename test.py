
from build_tag import *
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




class CaptionSampler(object):
    def __init__(self, args):
        self.args = args

        self.vocab = self.__init_vocab()
        self.tagger = self.__init_tagger()
        self.transform = self.__init_transform()
        self.data_loader = self.__init_data_loader(self.args.file_lits)
        self.model_state_dict = self.__load_mode_state_dict()
        self.extractor = self.__init_visual_extractor()
        self.diseaseClassifier = self.__init_mlc()
        self.gcn = self.__init_gcn()
        self.word_model = self.__init_word_word()
        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()


    def generate(self):
        self.extractor.eval()
        # self.mlc.eval()
        # self.co_attention.eval()
        # self.sentence_model.eval()
        # self.word_model.eval()
        progress_bar = tqdm(self.data_loader, desc='Generating')
        results = {}
        real = []
        pred = []
        pre2fuck = []
        real2fuck = []
        count1 = 0
        count2 = 0
        count3 = 0

        edge = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 9, 10, 10, 11,
                              11, 12, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,
                              17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20,
                              20, 20, 20],
                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 8, 7, 10, 11,
                              9, 11, 9, 10, 13, 12, 15, 16, 17, 18, 19, 20, 14, 16, 17, 18, 19, 20, 14, 15, 17, 18,
                              19, 20, 14, 15, 16, 18, 19, 20, 14, 15, 16, 17, 19, 20, 14, 15, 16, 17, 18, 20, 14,
                              15, 16, 17, 18, 19]], dtype=torch.long)
        edge = edge.cuda()

        for img_front, img_later, image_id, label, captions, _ in progress_bar:

            image_id = (image_id[0][0],image_id[1][0])
            img_front = self.__to_var(img_front)
            img_later = self.__to_var(img_later)

            # visual_features, avg_features = self.extractor.forward(images)
            front = self.extractor.forward(img_front)
            later = self.extractor.forward(img_later)

            front = front.view(front.shape[0], front.shape[1], -1)
            later = later.view(later.shape[0], later.shape[1], -1)

            f_global = torch.sum(front, dim=1) / 20
            l_global = torch.sum(later, dim=1) / 20

            f_global = f_global.unsqueeze(1)
            l_global = l_global.unsqueeze(1)

            front = torch.cat([f_global, front], dim=1)
            later = torch.cat([l_global, later], dim=1)

            front = self.gcn.forward(front, edge)
            later = self.gcn.forward(later, edge)

            prev_hidden_states = torch.cat([front[:, 0, :].unsqueeze(1), later[:, 0, :].unsqueeze(1)], dim=2)
            vis_feature = torch.cat([front[:, 1:, :], later[:, 1:, :]], dim=2)

            tags = self.mlc.forward(vis_feature).squeeze(2)



            prev_hidden_states = torch.cat([front[:, 0, :].unsqueeze(1), later[:, 0, :].unsqueeze(1)], dim=2)
            sentence_states = None

            pred_sentences = {}
            real_sentences = {}
            for i in image_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}

            for i in range(self.args.s_max):



                vt, prev_hidden_states, sentence_states, stop, word_input = self.sentence_model(vis_feature,
                                                                                                prev_hidden_states,
                                                                                                sentence_states)

                stop = stop.squeeze(1)
                stop = torch.max(stop, 1)[1].unsqueeze(1)

                start_tokens = np.zeros((word_input.shape[0], 1))
                start_tokens[:, 0] = self.vocab('<start>')
                start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)

                sampled_ids = self.word_model.sample(word_input, start_tokens)
                sampled_ids = torch.tensor(sampled_ids).cuda()
                prev_hidden_states = prev_hidden_states
                sampled_ids = sampled_ids * stop

                # self._generate_cam(image_id, visual_features, alpha_v, i)
                ps = ''
                for id, array in zip(image_id, sampled_ids):
                    pred_sentences[id][i] = self.__vec2sent(array.cpu().detach().numpy())
                    ps = ps + pred_sentences[id][i]+' '
                    pre2fuck.append(pred_sentences[id][i])
                pred.append(ps)
                count1 += 1

            for id, array in zip(image_id, captions):
                rs = ''
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)
                    rs = rs + real_sentences[id][i]+' '
                    real2fuck.append(real_sentences[id][i])
                real.append(rs)
                count2 += 1

            for id, pred_tag, real_tag in zip(image_id, tags, label):
                count3 += 1
                results[id] = {
                    'Real Tags': self.tagger.inv_tags2array(real_tag),
                    'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy()),
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }
                # a = pred_sentences[:,id]
                # print(111)
        real = real[:366]
        with open('p.txt', 'a') as f:
            for e in range(len(pred)):
                f.write(pred[e])
                f.write('\n')
        with open('r.txt', 'a') as f:
            for e in range(len(real)):
                f.write(real[e])
                f.write('\n')

        self.__save_json(results)

    def __save_json(self, result):
        result_path = os.path.join(self.args.model_dir, self.args.result_path)
        print(result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f, indent=4)

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(os.path.join(self.args.model_dir, self.args.load_model_path))
            print("[Load Model-{} Succeed!]".format(self.args.load_model_path))
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_tagger(self):
        return Tag()

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_data_loader(self, file_list):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=False)
        return data_loader

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __init_visual_extractor(self):

        model = Visualmod.VisualFeatureExtractor()
        if self.model_state_dict is not None:
            print("Visual Extractor Loaded!")
            model.load_state_dict(self.model_state_dict['extractor'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_gcn(self):
        model = Gmod.GCN1(features_in=self.args.gcnin, features_out=self.args.gcnout)
        if self.model_state_dict is not None:
            print("fcn Loaded!")
            model.load_state_dict(self.model_state_dict['gcn'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_mlc(self):

        model = classification.mlc(in_features=self.args.sementic_features_dim, classes=self.args.classes)

        if self.model_state_dict is not None:
            print("MLC Loaded!")
            model.load_state_dict(self.model_state_dict['mlc'])

        if self.args.cuda:
            model = model.cuda()

        return model

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
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    # Path Argument
    parser.add_argument('--model_dir', type=str, default='D:/gcn/report_v4_models/v420210119-0750/')
    parser.add_argument('--image_dir', type=str, default='D:/Medical-Report-Generation-master/data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='D:/Medical-Report-Generation-master/data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--vocab_path', type=str, default='D:/Medical-Report-Generation-master/data/new_data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--file_lits', type=str, default='D:/gcn/test.txt',
                        help='the path for test file list')
    parser.add_argument('--load_model_path', type=str, default='train_best_loss.pth32.tar',
                        help='The path of loaded model')

    # transforms argument
    parser.add_argument('--resize', type=int, default=512,
                        help='size for resizing images')

    # CAM
    parser.add_argument('--cam_size', type=int, default=224)
    parser.add_argument('--generate_dir', type=str, default='cam')

    # Saved result
    parser.add_argument('--result_path', type=str, default='results',
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='debug',
                        help='the name of results')

    """
    Model argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='not using pretrained model when training')
    parser.add_argument('--load_visual_model_path', type=str,
                        default='')
    parser.add_argument('--visual_trained', action='store_true', default=True,
                        help='Whether train visual extractor or not')

    # MLC
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--load_mlc_model_path', type=str,
                        default='')
    parser.add_argument('--mlc_trained', action='store_true', default=True)

    # 图结构
    parser.add_argument('--gcnin', type=int, default=256)
    parser.add_argument('--gcnout', type=int, default=256)
    parser.add_argument('--load_gcn_model_path', type=str,
                        default='')

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v4')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--visual_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--load_sentence_model_path', type=str,
                        default='')
    parser.add_argument('--sentence_trained', action='store_true', default=True)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)
    parser.add_argument('--load_word_model_path', type=str,
                        default='')
    parser.add_argument('--word_trained', action='store_true', default=True)

    """
    Generating Argument
    """
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    parser.add_argument('--batch_size', type=int, default=2)

    # Loss function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print(args)

    sampler = CaptionSampler(args)
    sampler.generate()