import json, torch, re, os
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer
from tree_sitter import Language, Parser
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pattern_extractor.pattern_time import time_gen_pattern
from pattern_extractor.Pattern_reen import reen_gen_pattern
from config import Argument


args = Argument().args
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if tokenizer.pad_token_id is None:
    print('tokenizer.pad_token_id is None:')
    tokenizer.pad_token_id = 0
if tokenizer.pad_token_id != 0:
    print('tokenizer.pad_token_id != 0:')
    tokenizer.pad_token_id = 0
print("空位占位符：")
print(tokenizer.pad_token_id)

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


class DataSet(Dataset):
    def __init__(self, codes, label, model, decs):
        super(DataSet, self).__init__()
        self.label = label
        self.ids = []
        self.pos = []
        global drop_list
        global number_list 
        global func_list 

        for i in tqdm(codes, ncols=160, desc=decs, ascii=' >>>+'):
            number_list = []
            drop_list = []
            func_list = []
            
            ################# AST Slice #################
            code = code_slice(i)
            # print(code)

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
            #########################################
            # code_tokens = code_tokens[:512-2] 
            # source_tokens = [tokenizer.bos_token] +\
            #     code_tokens+[tokenizer.eos_token]  
            # position_idx = [i+tokenizer.pad_token_id +1 for i in range(len(source_tokens))] 
            # ==================================== #
            code_tokens = code_tokens[:512-2-5]      
            source_tokens = code_tokens+[tokenizer.eos_token]
            position_idx = [i+tokenizer.pad_token_id +1 for i in range(len(source_tokens)+6)]
            # ==================================== #
            # code_tokens = code_tokens[:512-2-pattern_lang]      
            # source_tokens = code_tokens+[tokenizer.eos_token]
            # position_idx = [i+tokenizer.pad_token_id +1 for i in range(len(source_tokens)+pattern_lang+1)]
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

            self.ids.append(out)
            self.pos.append(position_idx)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.ids[idx],
                torch.tensor(self.pos[idx]),
                torch.tensor(self.label[idx]))


def Dataset_process(args, model):
    def load_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            code = [json.loads(x)['contract'] for x in data]
            label = [int(json.loads(x)['label']) for x in data]
        return code, label
    
    p1=args.train_data_file
    p2=args.eval_data_file
    p3=args.test_data_file

    train_dataset = DataSet(load_data(p1)[0], load_data(p1)[1], model, decs='train_data')
    eval_dataset = DataSet(load_data(p2)[0], load_data(p2)[1], model, decs='eval_data')
    test_dataset = DataSet(load_data(p3)[0], load_data(p3)[1], model, decs='test_data')

    # train_sampler = SequentialSampler(train_dataset)
    # eval_sampler = SequentialSampler(eval_dataset)
    # test_sampler = SequentialSampler(test_dataset)

    # train_loader = DataLoader(dataset=train_sampler, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    # eval_loader = DataLoader(dataset=eval_sampler, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    # test_loader = DataLoader(dataset=test_sampler, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    return train_loader, eval_loader, test_loader
    # return eval_loader, test_loader