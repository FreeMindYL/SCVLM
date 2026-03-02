import torch
# torch.cuda.set_device(2)
import sys
from tree_sitter import Language, Parser
from tools import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tools import DFG_solidity, rise_cfg, get_cfg
import os, re
import random
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, AutoTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm



# ======================================== bert data process ========================================
# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):

    args = bert_args

    # remove comments
    # try:
    #     code = remove_comments_and_docstrings(code, lang)
    # except:
    #     pass
    # obtain dataflow
    if lang == "php":
        code = "<?php"+code+"?>"
    try:

        func = code.replace("\n", " ").replace('\t', ' ')
        func = re.sub(' +', ' ', func)
        func = func.replace(';', ';\n').replace('{', '{\n').replace('}', '}\n')

        if 'time' in str(args.train_data_file):
            # print('robert source code tran')
            code = func

        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')

        # tree = parser[0].parse(bytes(func, 'utf8'))
        # root_node = tree.root_node
        # tokens_index = tree_to_token_index(root_node)
        # code = func.split('\n')

        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            rise_cfg()
            DFG, _ = parser[1](root_node, index_to_code, {})

            old_dfg = DFG

            CFG_line = get_cfg()
            # print(CFG_line)
            # print('*'*30)
            # print(DFG)

        except:
            print('err')
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        # identify critical node in DFG
        critical_idx = []
        for id, e in enumerate(DFG):
            if e[0] == "call" and DFG[id+1][0] == "value":
            # if e[0] == "block" and DFG[id+1][0] == "timestamp":  # block.timestamp
                # critical_idx.append(DFG[id-1][1])
                # critical_idx.append(DFG[id+2][1])
                critical_idx.append(DFG[id][1])
                # print(DFG[id-1])
                # print(DFG[id])
                # print(DFG[id+1])
                # print(DFG[id+2])
                # print('*'*30)
        lines = []
        for index, code in index_to_code.items():
            if code[0] in critical_idx:
                line = index[0][0]
                lines.append(line)

        lines+=CFG_line

        lines = list(set(lines))
        for index, code in index_to_code.items():  # ((16, 13), (16, 27)): (52, 'approveAndCall')
            # print(index)
            # print(code)
            # print(index_to_code)
            if index[0][0] in lines:
                critical_idx.append(code[0])
        critical_idx = list(set(critical_idx))
        max_nums = 0
        cur_nums = -1
        while cur_nums != max_nums and cur_nums != 0:
            max_nums = len(critical_idx)
            for id, e in enumerate(DFG):
                if e[1] in critical_idx:
                    critical_idx += e[-1]
                for i in e[-1]:
                    if i in critical_idx:
                        critical_idx.append(e[1])
                        break
            critical_idx = list(set(critical_idx))
            # print(critical_idx)
            # print("-"*40)
            cur_nums = len(critical_idx)
        dfg = []
        for id, e in enumerate(DFG):
            if e[1] in critical_idx:
                dfg.append(e)
        dfg = sorted(dfg, key=lambda x: x[1])

        # Removing independent points
        indexs = set()
        for d in dfg:
            # print(d)
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in dfg:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []

    # if len(CFG_line) > 0 and len(old_dfg)<180 and len(dfg)<100:
    #     print(func)
    #     print("-"*30)
    #     print(dfg)
    #     print("-"*30)
    #     print(old_dfg)
    #     print('\n')

    # print(dfg)
    # print('*'*60)
    return code_tokens, dfg

# ======================================== bert data process end ========================================



# ======================================== llm data process ========================================
# 加interface ??
skip = ['constructor_definition', 'modifier_definition', 'fallback_receive_definition', 'using_directive', 'event_definition']   #, 'emit_statement'
def ast_process(root_node):
    if (len(root_node.children)==0 or root_node.type=='string' or root_node.type=='comment' or 'comment' in root_node.type):
        if root_node.type=='number_literal':
            number = str(root_node.text, encoding = 'utf-8')
            if len(number) > 4:
                number_list.append(number)
    else:
        for child in root_node.children:
            if child.type in skip:
                code = str(child.text, encoding = 'utf-8')
                drop_list.append(code)
                continue 
            else:
                if child.type == 'function_definition':
                    func_code = str(child.text, encoding = 'utf-8')
                    func_code = func_code.replace(';', ';\n').replace('{', '{\n').replace('}', '}\n')
                    func_code = func_code.split('\n')
                    func_list.append(func_code)
                ast_process(child)

def code_slice(code):
    func = code.replace("\n", " ").replace('\t', ' ')
    func = re.sub(' +', ' ', func)

    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    language = Language('./tools/my-languages.so', 'solidity')
    parser = Parser()
    parser.set_language(language)
    tree = parser.parse(bytes(func, 'utf8'))
    root_node = tree.root_node
    ast_process(root_node)
    return func

from pattern_extractor.pattern_time import time_gen_pattern
from pattern_extractor.Pattern_reen import reen_gen_pattern
from model import load_llm
from config import Argument

llm = load_llm("train")
llm_args = Argument().args
llm_tokenizer = AutoTokenizer.from_pretrained(llm_args.model_path)
if llm_tokenizer.pad_token_id is None:
    print('tokenizer.pad_token_id is None:')
    llm_tokenizer.pad_token_id = 0
if llm_tokenizer.pad_token_id != 0:
    print('tokenizer.pad_token_id != 0:')
    llm_tokenizer.pad_token_id = 0
print("llm空位占位符: ")
print(llm_tokenizer.pad_token_id)

def llm_data_process(code_in):
    global drop_list
    global number_list 
    global func_list 

    number_list = []
    drop_list = []
    func_list = []

    tokenizer = llm_tokenizer
    model = llm
    
    ################# AST Slice #################
    code = code_slice(code_in)
    # print(code)
    # print("------------")

    if len(drop_list) > 0:
        for drop_code in drop_list:
            code=code.replace(drop_code, ' ')
    # print(drop_list)

    if len(number_list) > 0:
        for value in number_list:
            pattern = r'\b{}\b'.format(value)
            code = re.sub(pattern, "NUM", code)
    # print(number_list)
    # print(code)
    # print("==============")
    
    ##############################################

    # patterns = time_gen_pattern(code, func_list)
    patterns = reen_gen_pattern(code, func_list)

    # print(patterns)
    tokenizer.pad_token_id=0
    # print(tokenizer.pad_token_id)

    pattern_lang = len(patterns)
    patterns = [tokenizer.bos_token] + patterns
    patterns_embed = []
    for pattern in patterns:
        x = tokenizer.tokenize(pattern)
        pattern_ids = tokenizer.convert_tokens_to_ids(x)   
        pattern_ids = torch.tensor(pattern_ids)
        embed=model.model.model.embed_tokens(pattern_ids)
        # print(pattern_ids)
        # embed=model.model.transformer.shared(pattern_ids)
        if embed.size()[0] > 1:
            embed = torch.mean(embed, dim=0, keepdim=True)
        
        patterns_embed.append(embed)

    patterns_embed = torch.cat(patterns_embed, dim=0) 
    code_tokens = tokenizer.tokenize(code) 
    # code_tokens = tokenizer.tokenize(code_in) 
    #########################################
    # code_tokens = code_tokens[:512-2] 
    # source_tokens = [tokenizer.bos_token] +\
    #     code_tokens+[tokenizer.eos_token]  
    # position_idx = [i+tokenizer.pad_token_id +1 for i in range(len(source_tokens))]    # 无标签
    # ==================================== #
    code_tokens = code_tokens[:512-2-5]      
    source_tokens = code_tokens+[tokenizer.eos_token]
    position_idx = [i+tokenizer.pad_token_id +1 for i in range(len(source_tokens)+6)]   # 正常
    # ==================================== #
    # code_tokens = code_tokens[:512-2-pattern_lang]      
    # source_tokens = code_tokens+[tokenizer.eos_token]
    # position_idx = [i+tokenizer.pad_token_id +1 for i in range(len(source_tokens)+pattern_lang+1)]    # 标签不等长
    #########################################

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)   
    padding_length = 512-len(source_ids)-6
    # padding_length = 512-len(source_ids)-pattern_lang-1
    # padding_length = 512-len(source_ids)
    position_idx += [tokenizer.pad_token_id]*padding_length
    source_ids += [tokenizer.pad_token_id]*padding_length
    source_ids = torch.tensor(source_ids)

    # print(source_ids)
    # print(position_idx)


    code_embed=model.model.model.embed_tokens(source_ids)
    # code_embed=model.model.transformer.shared(source_ids)
    out = torch.cat((patterns_embed, code_embed), dim=0)
    # out = code_embed

    # self.ids.append(out)
    # self.pos.append(position_idx)
    return out, position_idx

# ======================================== llm data process end ========================================


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens_1,
                 input_ids_1,
                 position_idx_1,
                 dfg_to_code_1,
                 dfg_to_dfg_1,
                 llm_ids, 
                 llm_pos,
                 label,
                 url1
                 ):
        # The code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1
        self.llm_ids = llm_ids
        self.llm_pos = llm_pos

        # label
        self.label = label
        self.url1 = url1




from config import Bertarg, Output, Logger
bert_args = Bertarg().args
logger = Logger().logger
sys.stdout = Output()
dfg_function = {
    'solidity': DFG_solidity
}
# load parsers
import warnings
warnings.simplefilter('ignore', FutureWarning)
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('./tools/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser
bert_tokenizer = RobertaTokenizer.from_pretrained(bert_args.tokenizer_name)


def convert_examples_to_features(item):
    # source
    url1, label, tokenizer, args, cache, url_to_code = item
    parser = parsers['solidity']

    for url in [url1]:
        if url not in cache:
            func = url_to_code[url]

            # ================= llm data =================
            llm_ids, llm_pos = llm_data_process(func)
            # ================= ======== =================


            # ================= bert data =================
            # extract data flow
            code_tokens, dfg = extract_dataflow(func, parser, 'solidity')
            code_tokens = [tokenizer.tokenize(
                '@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i-1][1],
                                  ori2cur_pos[i-1][1]+len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            code_tokens = code_tokens[:args.code_length+args.data_flow_length -
                                      3-min(len(dfg), args.data_flow_length)][:512-3]
            source_tokens = [tokenizer.cls_token] + \
                code_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id +
                            1 for i in range(len(source_tokens))]
            dfg = dfg[:args.code_length +
                      args.data_flow_length-len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [tokenizer.unk_token_id for x in dfg]
            padding_length = args.code_length + \
                args.data_flow_length-len(source_ids)
            position_idx += [tokenizer.pad_token_id]*padding_length
            source_ids += [tokenizer.pad_token_id]*padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1]+([reverse_index[i]
                                    for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]
            cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, llm_ids, llm_pos

    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1, llm_ids, llm_pos = cache[url1]
    return InputFeatures(source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1, llm_ids, llm_pos,
                         label, url1)


class AllDataset(Dataset):
    def __init__(self, desc, file_path='train'):
        self.examples = []
        self.args = bert_args
        index_filename = file_path

        tokenizer = bert_tokenizer

        # load index
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code = {}
        data = []
        cache = {}
        with open(index_filename, 'r') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['contract']
                url1 = js['idx']
                url1 = int(url1)
                label = js['label']
                label = int(label)
                data.append((url1, label, tokenizer, self.args, cache, url_to_code))

        # convert example to input features
        # self.examples = [convert_examples_to_features(x) for x in tqdm(data, total=len(data), ncols=160, ascii='/_')]
        self.examples = [convert_examples_to_features(x) for x in tqdm(data, ncols=160, desc=desc, ascii=' >>>+')]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask_1 = np.zeros((self.args.code_length+self.args.data_flow_length,
                                self.args.code_length+self.args.data_flow_length), dtype=np.bool_)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_1])
        # sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_1):
            if a < node_index and b < node_index:
                attn_mask_1[idx+node_index, a:b] = True
                attn_mask_1[a:b, idx+node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a+node_index < len(self.examples[item].position_idx_1):
                    attn_mask_1[idx+node_index, a+node_index] = True

        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].label),
                self.examples[item].llm_ids,
                torch.tensor(self.examples[item].llm_pos)
                )

def All_dataset_process(llm_args, bert_args, model_type):
    
    p1=llm_args.train_data_file
    p2=llm_args.eval_data_file
    p3=llm_args.test_data_file

    # train_dataset = AllDataset(desc='train_data', file_path=p1)
    eval_dataset = AllDataset(desc='eval_data', file_path=p2)
    test_dataset = AllDataset(desc='test_data', file_path=p3)

    if (model_type == "bert"):
        batch_size = bert_args.train_batch_size
    else:
        batch_size = llm_args.batch_size

    print("batch_size:"+ str(batch_size))

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    # eval_loader = DataLoader(dataset=eval_dataset, batch_size= 1 , shuffle=False, drop_last=True, num_workers=0)
    # test_loader = DataLoader(dataset=test_dataset, batch_size= 1 , shuffle=False, drop_last=True, num_workers=0)
    # return train_loader, eval_loader, test_loader
    return eval_loader, test_loader

