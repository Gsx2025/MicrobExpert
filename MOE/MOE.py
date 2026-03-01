import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import pandas as pd
import gc
from collections import Counter
# 指定要使用的 GPU 设备
torch.cuda.set_device(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 设置随机种子以保证可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

set_seed(26)

# 加载专家分词器和模型
def load_expert_models():
    tokenizer1 = AutoTokenizer.from_pretrained("/home/stu_6/GSX/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                               do_lower_case=True)
    tokenizer2 = AutoTokenizer.from_pretrained("/home/stu_6/GSX/Bio_ClinicalBERT", do_lower_case=True)
    tokenizer3 = AutoTokenizer.from_pretrained("/home/stu_6/GSX/bert-base-uncased", do_lower_case=True)
    tokenizer4 = AutoTokenizer.from_pretrained("/home/stu_6/GSX/BioLinkBERT-base", do_lower_case=True)
    tokenizer5 = AutoTokenizer.from_pretrained("/home/stu_6/GSX/BioMegatron345mUncased", do_lower_case=True)

    model1 = AutoModel.from_pretrained("/home/stu_6/GSX/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model2 = AutoModel.from_pretrained("/home/stu_6/GSX/Bio_ClinicalBERT")
    model3 = AutoModel.from_pretrained("/home/stu_6/GSX/bert-base-uncased")
    model4 = AutoModel.from_pretrained("/home/stu_6/GSX/BioLinkBERT-base")
    model5 = AutoModel.from_pretrained("/home/stu_6/GSX/BioMegatron345mUncased")

    model1.to(device)
    model2.to(device)
    model3.to(device)
    model4.to(device)
    model5.to(device)

    return [tokenizer1, tokenizer2, tokenizer3, tokenizer4, tokenizer5], [model1, model2, model3, model4, model5]

# 定义类别和输入维度
num_classes = 4
input_dim = 768  # 使用 BERT 的 pooler_output
hidden_dim = 768  # 专家模型的输入维度
mapping = {"positive": 0, "negative": 1, "relate": 2, "NA": 3}
reverse = {v: k for k, v in mapping.items()}

# 定义门控网络
class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(TopKGating, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),  # 隐藏层
            nn.ReLU(),
            nn.Linear(64, num_experts)  # 输出层
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化权重
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        gating_scores = self.gate(x)
        probabilities = F.softmax(gating_scores, dim=1)  # 应用softmax函数
        #if torch.rand(1).item() < 0.05:
            #print("Gating softmax probs (first sample):", probabilities[0].detach().cpu().numpy())
        top_k_prob, top_k_index = torch.topk(probabilities, 2, dim=1)  # 获取概率最高的2个专家及其概率
        return top_k_index, top_k_prob

# 定义专家类
class Expert(nn.Module):
    def __init__(self, model, num_classes, tokenizer):
        super(Expert, self).__init__()
        self.bert = model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.tokenizer = tokenizer  # 添加分词器为类的属性

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


class MoEModel(nn.Module):
    def __init__(self, expert_models, num_classes, input_dim, tokenizers):
        super(MoEModel, self).__init__()
        self.gating_network = TopKGating(input_dim, len(expert_models))
        self.num_classes = num_classes
        self.tokenizers = tokenizers
        self.experts = nn.ModuleList(
            [Expert(model, num_classes, tokenizer) for model, tokenizer in zip(expert_models, tokenizers)])

    def forward(self, word2vec_features, input_ids, attention_mask):
        top_k_index, top_k_prob = self.gating_network(word2vec_features)
        outputs = torch.zeros(input_ids.shape[0], self.num_classes, device=input_ids.device)

        # 选择前两个专家的输出
        for i in range(2):
            expert_outputs = []
            for j in range(input_ids.shape[0]):
                expert_idx = top_k_index[j, i].item()
                logits = self.experts[expert_idx](input_ids[j, i, :].unsqueeze(0), attention_mask[j, i, :].unsqueeze(0))
                expert_outputs.append(logits.squeeze(0))

            expert_outputs = torch.stack(expert_outputs)
            expert_prob = top_k_prob[:, i].unsqueeze(-1)
            outputs += expert_outputs * expert_prob

        # 确保概率归一化
        total_prob = torch.sum(top_k_prob, dim=1).unsqueeze(-1)
        outputs /= total_prob
        return outputs


# Data processing
def encode_data(tokenizer, passages, questions, max_length=512):
    inputs = tokenizer(passages, questions, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return inputs
    
def generate_bert_cls_features(model, tokenizer, passages, questions):
    model.eval()
    inputs = tokenizer(passages, questions, padding='max_length', truncation=True,
                       max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output

# 预测函数
def predict_for_moe(model, passage, question, device, tokenizers, expert_models, expert_select_counter_test):
    model.eval()
    # 替换 gating 特征为 BERT-base [CLS]
    gate_features = generate_bert_cls_features(expert_models[3], tokenizers[3], [passage], [question])

    with torch.no_grad():
        top_k_indices, top_k_probs = model.gating_network(gate_features)
        expert_select_counter_test.update(top_k_indices[0].tolist())
        logits = torch.zeros((1, model.num_classes), device=device)

        for i in range(2):
            expert_index = top_k_indices[0, i]
            expert_prob = top_k_probs[0, i]
            selected_tokenizer = tokenizers[expert_index.item()]
            inputs = encode_data(selected_tokenizer, [passage], [question], max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            selected_expert = model.experts[expert_index.item()]
            expert_logits = selected_expert(input_ids, attention_mask)
            logits += expert_logits * expert_prob

    probabilities = torch.softmax(logits, dim=-1)
    predicted_index = probabilities.argmax(dim=1).item()
    predicted_label = reverse[predicted_index]
    return predicted_label

#训练函数
def train_model(model, train_data_df, tokenizers, expert_models, expert_select_counter_train, num_epochs, batch_size, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_data_df) // batch_size * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for start_index in range(0, len(train_data_df), batch_size):
            end_index = min(start_index + batch_size, len(train_data_df))
            batch = train_data_df.iloc[start_index:end_index]
            current_batch_size = len(batch)
            labels = torch.tensor(batch['RELATION'].apply(lambda x: x[1] if isinstance(x, tuple) else x).values, dtype=torch.long).to(device)
            passages = batch['EVIDENCE'].values.tolist()
            questions = batch['QUESTIONS'].values.tolist()

            gate_features = generate_bert_cls_features(expert_models[3], tokenizers[3], passages, questions)
            top_k_index, top_k_prob = model.gating_network(gate_features)
            for idx in top_k_index:
                expert_select_counter_train.update(idx.tolist())

            all_input_ids = []
            all_attention_masks = []

            for idx in range(current_batch_size):
                passage = passages[idx]
                question = questions[idx]
                input_ids_list = []
                attention_mask_list = []
                for expert_idx in top_k_index[idx]:
                    selected_tokenizer = tokenizers[expert_idx.item()]
                    inputs = encode_data(selected_tokenizer, [passage], [question], max_length=512)
                    input_ids_list.append(inputs['input_ids'].squeeze(0))
                    attention_mask_list.append(inputs['attention_mask'].squeeze(0))
                all_input_ids.append(torch.stack(input_ids_list))
                all_attention_masks.append(torch.stack(attention_mask_list))

            input_ids = torch.stack(all_input_ids).to(device)
            attention_mask = torch.stack(all_attention_masks).to(device)

            outputs = model(gate_features, input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        print(f" Epoch Training Loss: {total_train_loss / (len(train_data_df) // batch_size)}")
        print(f"Expert Usage After Epoch {epoch + 1} (Training):")
        for expert_id, count in sorted(expert_select_counter_train.items()):
            print(f"Expert {expert_id}: {count} times")
        expert_select_counter_train.clear()

    return model

# === 主程序逻辑 ===
df = pd.read_csv('/home/stu_6/GSX/data-final/F-M-GSC_low_depth.csv')
df = df[['MICROBE', 'DIETARY', 'EVIDENCE', 'RELATION', 'QUESTIONS']]
df.RELATION = df.RELATION.fillna('NA')
for i in range(len(df)):
    df['RELATION'][i] = str(df['RELATION'][i])
    df['RELATION'][i] = mapping[df['RELATION'][i]]

kf = KFold(n_splits=5, random_state=42, shuffle=True)

CV_accuracy_array = []
CV_macro_avg_array = []
CV_weighted_avg_array = []
CV_precision_macro_array = []
CV_recall_macro_array = []
CV_precision_weighted_array = []
CV_recall_weighted_array = []

expert_select_counter_train = Counter()
expert_select_counter_test = Counter()

for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
    print(f"Fold {fold}")
    print("TRAIN:", train_index, "TEST:", test_index)

    train_data_df, dev_data_df = df.iloc[train_index], df.iloc[test_index]
    tokenizers, expert_models = load_expert_models()
    moe_model_i = MoEModel(expert_models, num_classes, input_dim, tokenizers).to(device)
    moe_model = train_model(moe_model_i, train_data_df, tokenizers, expert_models, expert_select_counter_train, 15, 2, 1e-5)

    preds = []
    true_labels = []
    for index, row in dev_data_df.iterrows():
        passage = row['EVIDENCE']
        question = row['QUESTIONS']
        label = row['RELATION']
        predicted_label = predict_for_moe(moe_model, passage, question, device, tokenizers, expert_models, expert_select_counter_test)
        preds.append(predicted_label)
        true_labels.append(label)

    str_true_labels = [reverse[label] for label in true_labels]
    report = classification_report(str_true_labels, preds, output_dict=True)

    print(classification_report(str_true_labels, preds))
    print("Expert Usage Summary (Prediction):")
    for expert_id, count in sorted(expert_select_counter_test.items()):
        print(f"Expert {expert_id}: {count} times")
    expert_select_counter_test.clear()


    CV_accuracy_array.append(report['accuracy'])
    CV_macro_avg_array.append(report['macro avg']['f1-score'])
    CV_precision_macro_array.append(report['macro avg']['precision'])
    CV_recall_macro_array.append(report['macro avg']['recall'])
    CV_weighted_avg_array.append(report['weighted avg']['f1-score'])
    CV_precision_weighted_array.append(report['weighted avg']['precision'])
    CV_recall_weighted_array.append(report['weighted avg']['recall'])

    del moe_model
    gc.collect()
    torch.cuda.empty_cache()

print("Overall Performance Metrics:")
print("Accuracy:", CV_accuracy_array)
print("Macro Avg F1-score:", CV_macro_avg_array)
print("Weighted Avg F1-score:", CV_weighted_avg_array)
print("Macro Avg Precision:", CV_precision_macro_array)
print("Macro Avg Recall:", CV_recall_macro_array)
print("Weighted Avg Precision:", CV_precision_weighted_array)
print("Weighted Avg Recall:", CV_recall_weighted_array)
