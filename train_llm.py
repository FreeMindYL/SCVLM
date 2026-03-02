import sys, random, os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
from config import Argument, Output, Logger, Bertarg
args = Argument().args
logger = Logger().logger
sys.stdout = Output()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

from tqdm import tqdm
from llm_dataload import Dataset_process
from all_dataload import All_dataset_process
from model import get_final_rs, load_llm, load_bert, get_my_model
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train(args, llm, bert,train_loader, eval_loader):
    model = llm

    args.max_steps = args.epochs*len(train_loader)
    args.save_steps = len(train_loader)//10
    args.warmup_steps = args.max_steps//5
    model.to(args.device)
    bert.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)   # weight dency

    print(args.lr)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_loader, total=len(train_loader), ncols=160, ascii=' >=')
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            # (inputs_ids, position_idx, labels) = [x.to(args.device) for x in batch]
            (bert_ids, bert_idx, bert_musk, labels, inputs_ids, position_idx) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = get_final_rs(args, model, bert, bert_ids, bert_idx, bert_musk, inputs_ids, position_idx, labels, 0, 0)  

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm) 

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            avg_loss = round(
                np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
            

            if global_step % args.save_steps == 0:
                results = evaluate(args, model, bert, eval_loader, 0, 0, "cmd")

                # Save model checkpoint
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  "+"*"*20)
                    logger.info("  Best f1:%s", round(best_f1, 4))
                    logger.info("  "+"*"*20)

                    out = '模型保存至->'+args.save_model_path
                    out_len = len(out)+12
                    print("—"*(out_len+12))
                    print("|"+" "*5+ out +" "*5+"|")
                    print("—"*(out_len+12))

                    model.save_pretrained(args.save_model_path)
                    logger.info("Saving model checkpoint to best_eval_checkpoint")    

                      
def evaluate(args, llm, bert, eval_loader, q_coeff, v_coeff, output_type):
    # Eval!
    if output_type == "cmd":
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", args.batch_size)

    model = llm

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    import time
    num = 0
    start = time.time()
    for batch in eval_loader:
        # print(batch)
        # (inputs_ids, position_idx, labels) = [x.to(args.device) for x in batch]
        # (_, _, _, labels, inputs_ids, position_idx) = [x.to(args.device) for x in batch]
        (bert_ids, bert_idx, bert_musk, labels, inputs_ids, position_idx) = [x.to(args.device) for x in batch]
          
        with torch.no_grad():
            # lm_loss, logit = forward(args, model, inputs_ids, position_idx, labels)
            lm_loss, logit = get_final_rs(args, model, bert, bert_ids, bert_idx, bert_musk, inputs_ids, position_idx, labels, q_coeff, v_coeff)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.to(torch.float).cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

        num +=1
        if num == 100:
            end = time.time()
            time_out = end - start
            time_1 = time_out/100
            time_10 = time_out/10

    

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0

    y_preds = logits[:, 1] > best_threshold
    accuracy = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds, average='macro')
    precision = precision_score(y_trues, y_preds, average='macro')
    f1 = f1_score(y_trues, y_preds, average='macro')
    result = {
        "eval_accuracy": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }
    if output_type == "cmd":
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    print("平均1条时间: "+ str(time_1))
    print("平均10条时间: "+ str(time_10))

    return result


# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':
    # set_seed(args)

    # ==================== load date ====================
    # model = load_llm('train')
    # train_loader, eval_loader, test_loader = Dataset_process(args, model)
    # train_loader, eval_loader, test_loader = All_dataset_process(args, Bertarg().args, "llm")
    eval_loader, test_loader = All_dataset_process(args, Bertarg().args, "llm")




    # ==================== train ====================
    # model = load_llm('train')
    bert = load_bert("train", "reen")
    bert.to(args.device)
    # model = get_my_model(model, model)
    # train(args, model, bert, train_loader, eval_loader)




    # ==================== test ====================
    model = load_llm('test')
    add_ertra_model = True
    # add_ertra_model = False


    if add_ertra_model:
        q_coeff = 1
        v_coeff = 0
        model = get_my_model(model, model)
        model.to(args.device)
        evaluate(args, model, bert, test_loader, q_coeff, v_coeff, "cmd")
    # if add_ertra_model:
    #     model = get_my_model(model, model)
    #     model.to(args.device)
    #     bert.to(args.device)
    #     best_q_coeff = 0
    #     best_v_coeff = 0
    #     best_f1 = 0
    #     best_rs = {}
    #     step = 1
    #     for q_coeff in tqdm(np.arange(0, 5, 0.5), ncols=160, desc="search best coeff"):
    #         for v_coeff in tqdm(np.arange(0, 5, 0.5), ncols=130, desc="step "+str(step), ascii="_\\", leave=False):
    #             rs = evaluate(args, model, bert, test_loader, q_coeff, v_coeff, "return")
    #             f1 = rs["eval_f1"]
    #             if f1 > best_f1:
    #                 best_f1 = f1
    #                 best_rs = rs
    #                 best_q_coeff = q_coeff
    #                 best_v_coeff = v_coeff
    #         step += 1
    #     print("best result:")
    #     for key in sorted(best_rs.keys()):
    #         print("  {} = {}".format(key, round(best_rs[key], 4)))
    #     print("best q: "+str(best_q_coeff)+"\n"+"best v:"+str(best_v_coeff))
    else:
        bert = None
        model.to(args.device)
        evaluate(args, model, bert, test_loader, 0, 0, "cmd")







