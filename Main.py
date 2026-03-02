import torch, random, os, sys, gc
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())
# torch.cuda.empty_cache()
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tqdm import tqdm
import numpy as np
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from train_roberta import TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from llm_dataload import Dataset_process
from model import get_final_rs, load_llm
from model import Model


def load_roberta_mix_data():
    from config import Bertarg
    args = Bertarg().args
    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config)
    model = Model(model, config, tokenizer, args)
    checkpoint_prefix = os.path.join(args.output_dir, 'model.bin')
    output_dir = checkpoint_prefix

    # cpu = torch.device('cpu')
    model.load_state_dict(torch.load(output_dir,map_location={'cuda:1':'cuda:0'}))
    # model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    # model.to('cpu')
    # train_dataset = TextDataset(
    #     tokenizer, args, file_path=args.train_data_file)
    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(
    #                         train_dataset, 
    #                         sampler=train_sampler, 
    #                         batch_size=args.train_batch_size, 
    #                         num_workers=4)
    # x_train = []
    # y_train = []
    # bar = tqdm(train_dataloader, total=len(train_dataloader), ncols=160, ascii=' _/', desc='load graph train')
    # for batch in bar:
    #     (inputs_ids_1, position_idx_1, attn_mask_1,labels) = [x.to(args.device) for x in batch]
    #     with torch.no_grad():
    #         lm_loss, logit = model(inputs_ids_1, position_idx_1, attn_mask_1, labels)
    #         x_train.append(logit.to(torch.float).cpu().numpy())
    #         y_train.append(labels.cpu().numpy())
    # x_train = np.concatenate(x_train, 0)
    # y_train = np.concatenate(y_train, 0)
    

    test_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)
    x_test = []
    y_test = []
    bar = tqdm(test_dataloader, total=len(test_dataloader), ncols=160, ascii=' _/', desc='load graph test')

    print(args.device)

    for batch in bar:
        (inputs_ids_1, position_idx_1, attn_mask_1,labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(inputs_ids_1, position_idx_1, attn_mask_1, labels)
            x_test.append(logit.to(torch.float).cpu().numpy())
            y_test.append(labels.cpu().numpy())
    x_test = np.concatenate(x_test, 0)
    y_test = np.concatenate(y_test, 0)

    # return x_train, y_train, x_test, y_test
    return x_test, y_test


def load_llm_mix_data():
    model = load_llm('test')
    # train_loader, eval_loader, test_loader = Dataset_process(args, model)
    eval_loader, test_loader = Dataset_process(args, model)
    model.to(args.device)

    # x_train = []
    # y_train = []
    # bar = tqdm(train_loader, total=len(train_loader), ncols=160, ascii=' _/', desc='load seq train')
    # for batch in bar:
    #     (inputs_ids, position_idx, labels) = [x.to(args.device) for x in batch]
    #     # (inputs_ids, position_idx, labels) = [x for x in batch]
    #     with torch.no_grad():
    #         lm_loss, logit = forward(args, model,inputs_ids, position_idx, labels)
    #         x_train.append(logit.to(torch.float).cpu().numpy())
    #         y_train.append(labels.cpu().numpy())
    # x_train = np.concatenate(x_train, 0)
    # y_train = np.concatenate(y_train, 0)
    
    x_test = []
    y_test = []
    bar = tqdm(test_loader, total=len(test_loader), ncols=160, ascii=' _/', desc='load seq test')
    for batch in bar:
        (inputs_ids, position_idx, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = get_final_rs(args, model,inputs_ids, position_idx, labels)
            x_test.append(logit.to(torch.float).cpu().numpy())
            y_test.append(labels.cpu().numpy())
    x_test = np.concatenate(x_test, 0)
    y_test = np.concatenate(y_test, 0)

    # return x_train, y_train, x_test, y_test
    return x_test, y_test


if __name__ == "__main__":
    from config import Argument, Output, Logger
    args = Argument().args
    logger = Logger().logger
    sys.stdout = Output()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)    

    # graph_x_train, graph_y_train, graph_x_test, graph_y_test = load_roberta_mix_data() 
    graph_x_test, graph_y_test = load_roberta_mix_data() 

    # print(graph_x_train)

    # seq_x_train, seq_y_train, seq_x_test, seq_y_test = load_llm_mix_data()
    seq_x_test, seq_y_test = load_llm_mix_data()

    # ###################### voting data ######################33
    # col1 = seq_x_test[:, 1].unsqueeze(1) 
    col1 = seq_x_test[:, 1]
    col2 = graph_x_test[:, 1] 

    # col3 = seq_x_test[:, 1]
    # col4 = graph_x_test[:, 1]

    # test_data = torch.cat((col1, col2), dim=1)    
    # test_data = np.hstack((col1, col2))
    test_label = seq_y_test 
    # ###################### ###### ######################33

    # ###################### stacking method ######################33
    # col1 = seq_x_train[:, 1:2]
    # col2 = graph_x_train[:, 1:2]
    # train_data = np.hstack((col2, col1))
    # train_label = seq_y_train 

    # col1 = seq_x_test[:, 1:2]
    # col2 = graph_x_test[:, 1:2]
    # test_data = np.hstack((col2, col1))
    # test_label = seq_y_test 

    # import xgboost as xgb
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.model_selection import GridSearchCV
    # from xgboost import XGBRegressor, XGBClassifier
    # from sklearn import metrics
    # print('mix train ...')
    # gbc.fit(train_data, train_label)
    # print('run mix test ...')
    # print(test_data)
    # y_preds = gbc.predict(test_data)  # <<<<<<=====================
    # y_preds = gbc.predict_proba(test_data)
    # print(y_preds)
    # ###################### ############# ######################33


    # prob = 0.65*col1 + 0.35*col2    # reen
    prob = 0.65*col1 + 0.35*col2
    # prob = col2

    # prob0 = 0.5*col2 + 0.5*col1
    # prob1 = 0.5*col3 + 0.5*col4
    # print(prob0)
    # print(prob1)

    # prob = col1
    y_preds = prob > 0.5
    # y_preds = y_preds[:, 1] > 0.7
    # y_preds = prob0 < prob1
    # y_preds = y_preds[:, 0] < y_preds[:, 1]

    # 07/26/2024 14:49:49 - INFO - config -   eval_f1 = 0.9457  0.71  / 0.933 0.72 / 0.9457 0.7
    # print(test_label)
    # print(y_preds)


    accuracy = accuracy_score(test_label, y_preds)
    recall = recall_score(test_label, y_preds, average='macro')
    precision = precision_score(test_label, y_preds, average='macro')
    f1 = f1_score(test_label, y_preds, average='macro')
    result = {
        "eval_accuracy": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))                                                                                                                                                              


    