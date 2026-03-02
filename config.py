from argparse import ArgumentParser
import logging, sys, os, torch


# Switch between reentrancy and timestamp dependency datasets by commenting/uncommenting below
train_data_file='./data/reen/train.jsonl'
eval_data_file='./data/reen/eval.jsonl'
test_data_file='./data/reen/test.jsonl'
# train_data_file='./data/time/train.jsonl'
# eval_data_file='./data/time/eval.jsonl'
# test_data_file='./data/time/test.jsonl'


class Argument(object):
    parser = ArgumentParser()
    parser.add_argument('--train_data_file', type=str, default=train_data_file)
    parser.add_argument('--eval_data_file', type=str, default=eval_data_file)
    parser.add_argument('--test_data_file', type=str, default=test_data_file)
    # Path to the LLM backbone (e.g., CodeGemma-7b). Download from HuggingFace.
    parser.add_argument('--model_path', type=str, default='./models/code_gemma')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints/llm')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=torch.device("cuda"))
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--n_gpu', type=int, default=torch.cuda.device_count())
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bert_input_length', type=int, default=256)
    parser.add_argument('--bert_graph_length', type=int, default=64)
    args = parser.parse_args()


class Bertarg(object):
    parser = ArgumentParser()
    parser.add_argument('--train_data_file', type=str, default=train_data_file)
    parser.add_argument('--eval_data_file', type=str, default=eval_data_file)
    parser.add_argument('--test_data_file', type=str, default=test_data_file)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/graph')
    # Path to UniXcoder model. Download from HuggingFace: microsoft/unixcoder-base
    parser.add_argument('--model_name_or_path', type=str, default='./models/unixcoder')
    parser.add_argument('--config_name', type=str, default='./models/unixcoder')
    parser.add_argument('--tokenizer_name', type=str, default='./models/unixcoder')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=torch.device("cuda"))
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--n_gpu', type=int, default=torch.cuda.device_count())
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--data_flow_length', type=int, default=256)
    parser.add_argument('--code_length', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    args = parser.parse_args()


os.makedirs('./logs', exist_ok=True)
logs_path = './logs/train.log'

class Output(object):
    def __init__(self, filename=logs_path):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()
    def flush(self):
        pass

class Logger(object):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    file_handler = logging.FileHandler(logs_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
