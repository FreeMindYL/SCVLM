import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (LlamaConfig,LlamaForSequenceClassification,CodeLlamaTokenizer,GemmaForSequenceClassification,
                          GemmaTokenizer,GemmaTokenizer,GemmaConfig,AutoTokenizer,T5Config,T5ForSequenceClassification,
                          Starcoder2ForSequenceClassification, Starcoder2Config,CodeGenForCausalLM,CodeGenConfig,
                          AutoConfig, AutoModelForSequenceClassification, RobertaConfig, 
                          RobertaForSequenceClassification, RobertaTokenizer
                          )
from peft import get_peft_model,LoraConfig,TaskType,PeftModel
from config import Argument, Bertarg
import peft
import bitsandbytes as bnb
from peft.tuners.lora import LoraLayer





# ======================================= Unix lora start =======================================
def get_unix_rs(args, llm, bert, bert_ids, bert_idx, bert_musk, inputs_ids, position_idx, labels):     # 还需要改变输入序列长度
    model = llm

    for param in model.parameters():
        param.requires_grad = False

    _,_,bert_out = bert(bert_ids, bert_idx, bert_musk, labels)

    # model.eval()
    # with torch.no_grad():
    model = pass_param(model, bert_out, bert.q_coeff, bert.v_coeff)

    output = model.model.model(inputs_embeds=inputs_ids, position_ids=position_idx)
        # output = model.model.transformer(inputs_embeds=inputs_ids, position_ids=position_idx)

    outputs = output[0]   
    logits=model.model.score(outputs)
    # logits=model.model.classification_head(outputs)
    
    sequence_lengths = torch.eq(position_idx, torch.tensor([0]).to(args.device)).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths.to(logits.device)

    batch_size = inputs_ids.shape[0]
    pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(pooled_logits.view(-1, model.num_labels), labels.view(-1))
    prob=F.softmax(pooled_logits, dim=1)

    # loss.requires_grad = True

    if labels is not None:                 
        return loss, prob
    else:
        return prob

def pass_param(model, bert_out, q_coeff, v_coeff):
    for name, module in model.model.model.layers.named_children():
        # temp = torch.zeros(bert_out.shape[0], 512, bert_out.shape[2], dtype = bert_out.dtype, device=args.device)
        # temp[:, :bert_out.shape[1], :] = bert_out
        # module.self_attn.q_proj.iii = temp
        # if (int(name)>20):
        module.self_attn.q_proj.extra_feature = bert_out
        module.self_attn.q_proj.extra_coeff = q_coeff
        module.self_attn.v_proj.extra_feature = bert_out
        module.self_attn.v_proj.extra_coeff = v_coeff

        # for idx, mod in module.self_attn.named_children():
        # #     if isinstance(mod, UnixLoraLinear):
        # #         mod.iii = bert_out
        # #         break
        #     # print(idx)
        #     # print(mod)
        # module.self_attn.v_proj.iii = bert_out
    return model


def get_my_model(model, model_fp):
    # print(model_fp)                                              // ir此处更改
    model.model = _replace_with_ours_lora_linear(model.model, model_fp=model_fp.model)
    return model


class UnixLoraLinear(peft.tuners.lora.Linear, LoraLayer):    # bnb.nn.Linear4bit
    def __init__(self, model):
        for key, value in model.__dict__.items():
            setattr(self, key, value)

        self.extra_feature = None
        self.extra_coeff = None
        # self.act = nn.ReLU()
        self.base_layer.compute_dtype = self.lora_A.default.weight.dtype
        # self.lora_default_A_scale = torch.nn.Parameter(torch.zeros([1], dtype=self.lora_A.default.weight.dtype).to(self.base_layer.weight.device), requires_grad=True)
        # self.lora_default_B_scale = torch.nn.Parameter(torch.zeros([1], dtype=self.lora_A.default.weight.dtype).to(self.base_layer.weight.device), requires_grad=True)
       
    
    def forward(self, x: torch.Tensor):
        if self.base_layer.bias is not None and self.base_layer.bias.dtype != x.dtype:
            self.base_layer.bias.data = self.base_layer.bias.data.to(x.dtype)
        # if getattr(self.base_layer.weight, 'quant_state', None) is None:
        #     print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
        inp_dtype = x.dtype
        if self.base_layer.compute_dtype is not None:
            x = x.to(self.base_layer.compute_dtype)
        bias = None if self.base_layer.bias is None else self.base_layer.bias.to(self.base_layer.compute_dtype)
        
        out = torch.nn.functional.linear(x, self.base_layer.weight, bias)
        out = out.to(inp_dtype)
        result = out
        
        if self.disable_adapters or self.active_adapter[0] not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter[0]] > 0:
            result = result.clone()
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)
                # =======================================
                if self.extra_feature is None:
                    x = self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x)) 
                else:
                    self.extra_feature = self.extra_feature.to(self.base_layer.compute_dtype)
                    x = self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x)) + \
                        self.extra_coeff*self.extra_feature.reshape([_ for _ in self.extra_feature.shape[:-1]] + \
                                                                    [self.lora_A[self.active_adapter[0]].out_features] + [-1]).mean(dim=-1)
                       
                x = self.lora_B[self.active_adapter[0]](x)
                # =======================================
                output = x.to(expected_dtype) * self.scaling[self.active_adapter[0]]
            else:
                x = self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x)) + self.lora_default_A_scale * x.reshape([_ for _ in x.shape[:-1]] + [self.lora_A[self.active_adapter[0]].out_features] + [-1]).mean(dim=-1)
                x = (self.lora_B[self.active_adapter[0]](x).reshape([_ for _ in x.shape] + [-1]) + self.lora_default_B_scale * x.unsqueeze(-1)).reshape([_ for _ in x.shape[:-1]] + [-1])
                output = x * self.scaling[self.active_adapter[0]]
            result += output

        return result

def _replace_with_ours_lora_linear(model, current_key_name=None, model_fp=None):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        # if isinstance(module, peft.tuners.lora.Linear4bit):   # 未更改代码
        if isinstance(module, peft.tuners.lora.Linear):   # 此处更改 去掉了对于是否压缩的判断
            model._modules[name] = UnixLoraLinear(model._modules[name])

        if len(list(module.children())) > 0:
            # print(module.children())
            if name in model_fp._modules:
                # print("4356565656")
                _ = _replace_with_ours_lora_linear(
                    module,
                    current_key_name, model_fp._modules[name]
                )
            else:
                _ = _replace_with_ours_lora_linear(
                    module,
                    current_key_name, None
                )
        current_key_name.pop(-1)
    return model
# ======================================== Unix lora end =========================================





# ======================================= bert model start =======================================
def load_bert(stage, date_type):
    args = Bertarg().args
    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config)
    model = Model(model, config, tokenizer, args)
    if (stage == "train"):
        return model

    # if (date_type == "reen"):
    # else:
    # print("\n"+checkpoint_prefix+"\n")
    # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    # model.load_state_dict(torch.load(output_dir), False)

    # print("已加载bert模型 | stage: "+ stage +" | date type: " + date_type)
    # return model



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)

        # self.q_coeff = torch.nn.Parameter(torch.zeros([1], requires_grad=True))
        # self.v_coeff = torch.nn.Parameter(torch.zeros([1], requires_grad=True))

        self.args=args
    
        
    def forward(self, inputs_ids,position_idx,attn_mask,labels=None): 
        bs,l=inputs_ids.size()
        # print(bs)
      
        inputs_ids=(inputs_ids.unsqueeze(1)).view(bs*1,l)
        position_idx=(position_idx.unsqueeze(1)).view(bs*1,l)  
        attn_mask=(attn_mask.unsqueeze(1)).view(bs*1,l,l)
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2) 
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        # outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]  
        # outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,position_ids=position_idx)[0]      # 0.8741  0.8741
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask)[0]    # 0.9093
        
        logits=self.classifier(outputs)
        prob=F.softmax(logits, dim=1)
        # prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob,outputs
        else:
            return prob
# ======================================= bert model end ========================================





# ======================================= llm model start =======================================
        
def get_final_rs(args, llm, bert, bert_ids, bert_idx, bert_musk, inputs_ids, position_idx, labels, q_coeff=None, v_coeff=None):     # 还需要改变输入序列长度
    model = llm

    if bert is None:
        pass
    else:
        bert.eval()
        with torch.no_grad():
            _,_,bert_out = bert(bert_ids, bert_idx, bert_musk, labels)
        model = pass_param(model, bert_out, q_coeff, v_coeff)

    output = model.model.model(inputs_embeds=inputs_ids, position_ids=position_idx)
    # output = model.model.transformer(inputs_embeds=inputs_ids, position_ids=position_idx)

    outputs = output[0]   
    logits=model.model.score(outputs)
    # logits=model.model.classification_head(outputs)
    
    sequence_lengths = torch.eq(position_idx, torch.tensor([0]).to(args.device)).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths.to(logits.device)

    batch_size = inputs_ids.shape[0]
    pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(pooled_logits.view(-1, model.num_labels), labels.view(-1))
    prob=F.softmax(pooled_logits, dim=1)

    if labels is not None:                 
        return loss, prob
    else:
        return prob



def load_llm(stage):
    args = Argument().args
    # config = GemmaConfig.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)  # Starcoder2Config
    config.problem_type = "single_label_classification"
    config._attn_implementation = "flash_attention_2"
    # config.vocab_size, config.d_model
    # print(config.vocab_size)
    # print(config.d_model)
    peft_config = LoraConfig(
        peft_type="LORA",
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["W_pack", "o_proj"]
    )

    if stage == 'train':
        # model = GemmaForSequenceClassification.from_pretrained(               # Starcoder2ForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
                                    args.model_path, 
                                    config = config,
                                    torch_dtype=torch.bfloat16,
                                    low_cpu_mem_usage = True
                                    )           
        model = get_peft_model(model, peft_config)
        return model

    if stage == 'test':
        # model = GemmaForSequenceClassification.from_pretrained(
        model = AutoModelForSequenceClassification.from_pretrained(
                                    args.model_path,
                                    config = config, 
                                    torch_dtype = torch.bfloat16,
                                    low_cpu_mem_usage = True
                                    )
        
        # Load the trained LoRA adapter from the save path specified in config
        test_model_path = args.save_model_path
        print(f"Loading model from: {test_model_path}")
        model = PeftModel.from_pretrained(model, test_model_path)
        return model
# ======================================== bert model end ========================================
        


