import jieba
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
import warnings
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from tqdm import tqdm
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
from threading import Thread
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings('ignore')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if os.path.exists("C:\\Users\\DELL\\Desktop\\simhei.ttf"):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("已设置SimHei字体")
    else:
        print("使用默认字体")
except Exception as e:
    print(f"字体设置失败: {e}")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-chinese',
            cache_dir='./models',
            use_auth_token=False,
            proxies=None
        )
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SentimentBERTModel(nn.Module):
    """情感分析BERT模型"""

    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(SentimentBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits


class TopicBERTModel(nn.Module):
    """主题分类BERT模型"""

    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(TopicBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits


class BERTsentimentAnalyzer:
    def __init__(self, max_length=128, batch_size=16, learning_rate=2e-5):
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # 初始化模型
        self.sentiment_model = None
        self.topic_model = None

        # 模型文件路径
        self.sentiment_model_path = 'best_sentiment_model.pth'
        self.topic_model_path = 'best_topic_model.pth'

        # 标签映射
        self.sentiment_labels = {0: '负面', 1: '中性', 2: '正面'}
        self.topic_labels = {0: '产品评价', 1: '客户服务', 2: '价格反馈', 3: '使用体验'}

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 情感关键词（用于规则修正）
        self.negative_boost_words = {
            '差', '坏', '垃圾', '失望', '讨厌', '不好', '糟糕', '坑', '贵', '慢',
            '难用', '卡顿', '故障', '问题', '缺陷', '不足', '差劲', '垃圾', '骗人',
            '坑爹', '垃圾货', '劣质', '破烂', '废物', '没用', '浪费时间', '后悔',
            '别买', '不推荐', '上当', '被骗', '垃圾产品', '质量差', '服务差', '很差',
            '非常差', '极差', '太差', '最差', '不好用', '难用', '垃圾服务'
        }

        self.positive_boost_words = {
            '好', '棒', '满意', '喜欢', '不错', '优秀', '推荐', '超值', '舒服',
            '完美', '惊喜', '赞', '漂亮', '流畅', '便捷', '快速', '专业', '值得',
            '物超所值', '性价比高', '很好', '非常好', '特别棒', '强烈推荐', '很棒',
            '非常好', '极其满意', '非常满意', '特别好', '非常不错'
        }

        self.strong_negative_words = {'垃圾', '骗人', '上当', '别买', '不推荐', '后悔', '差劲', '劣质', '坑爹', '废物'}
        self.strong_positive_words = {'完美', '惊喜', '强烈推荐', '物超所值', '特别棒', '极其满意', '非常满意'}
        self.negation_words = {'不', '没', '无', '未', '别', '莫', '勿', '休', '免', '非'}

        self.special_negation_patterns = {
            '不太差': 2, '不算差': 2, '不算坏': 2, '不糟糕': 2, '不差': 2,
            '不坏': 2, '不难用': 2, '不慢': 2, '不贵': 2, '不太贵': 2, '不算贵': 2,
        }
        # 训练进度回调函数
        self.training_callback = None

    def set_training_callback(self, callback):
        """设置训练进度回调函数"""
        self.training_callback = callback

    def check_models_exist(self):
        """检查模型文件是否存在"""
        return os.path.exists(self.sentiment_model_path) and os.path.exists(self.topic_model_path)

    def load_models(self):
        try:
            # 初始化模型结构
            self.sentiment_model = SentimentBERTModel(num_classes=3)
            self.topic_model = TopicBERTModel(num_classes=4)

            # 加载模型权重
            self.sentiment_model.load_state_dict(torch.load(self.sentiment_model_path, map_location=self.device))
            self.topic_model.load_state_dict(torch.load(self.topic_model_path, map_location=self.device))

            # 将模型移动到设备
            self.sentiment_model.to(self.device)
            self.topic_model.to(self.device)

            print("模型加载成功!")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def save_models(self):
        """保存模型到文件"""
        try:
            if self.sentiment_model is not None:
                torch.save(self.sentiment_model.state_dict(), self.sentiment_model_path)
                print(f"情感模型已保存到: {self.sentiment_model_path}")

            if self.topic_model is not None:
                torch.save(self.topic_model.state_dict(), self.topic_model_path)
                print(f"主题模型已保存到: {self.topic_model_path}")

            return True
        except Exception as e:
            print(f"模型保存失败: {e}")
            return False

    def preprocess_text(self, text):
        """文本预处理"""
        if not isinstance(text, str):
            return ""
        # 去除特殊字符和数字，保留中文和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5，。！？]', ' ', text)
        return text.strip()

    def _create_extended_dataset(self):
        """创建扩展的示例数据集"""
        extended_data = [
            # 产品评价 - 正面
            {"text": "这个产品质量很好，做工精细，用料扎实", "sentiment": 2, "topic": 0},
            {"text": "外观设计漂亮，颜色搭配很时尚", "sentiment": 2, "topic": 0},
            {"text": "功能齐全，操作简单易懂", "sentiment": 2, "topic": 0},
            {"text": "性能稳定，运行流畅不卡顿", "sentiment": 2, "topic": 0},

            # 产品评价 - 负面
            {"text": "质量一般，细节处理不够好", "sentiment": 0, "topic": 0},
            {"text": "设计普通，没有什么特色", "sentiment": 0, "topic": 0},
            {"text": "功能有限，不能满足需求", "sentiment": 0, "topic": 0},
            {"text": "性能不稳定，经常出现问题", "sentiment": 0, "topic": 0},

            # 客户服务 - 正面
            {"text": "客服态度很好，回答问题耐心", "sentiment": 2, "topic": 1},
            {"text": "售后服务到位，问题解决及时", "sentiment": 2, "topic": 1},
            {"text": "技术支持专业，解决方案有效", "sentiment": 2, "topic": 1},

            # 客户服务 - 负面
            {"text": "客服响应慢，等待时间太长", "sentiment": 0, "topic": 1},
            {"text": "售后服务差，推卸责任", "sentiment": 0, "topic": 1},
            {"text": "技术不专业，问题反复出现", "sentiment": 0, "topic": 1},

            # 价格反馈 - 正面
            {"text": "价格实惠，性价比很高", "sentiment": 2, "topic": 2},
            {"text": "物超所值，质量对得起价格", "sentiment": 2, "topic": 2},
            {"text": "促销活动很给力，价格优惠", "sentiment": 2, "topic": 2},

            # 价格反馈 - 负面
            {"text": "价格偏贵，性价比不高", "sentiment": 0, "topic": 2},
            {"text": "定价不合理，同价位有更好选择", "sentiment": 0, "topic": 2},

            # 使用体验 - 正面
            {"text": "使用方便，上手很快", "sentiment": 2, "topic": 3},
            {"text": "体验流畅，感觉很舒适", "sentiment": 2, "topic": 3},
            {"text": "操作简单，学习成本低", "sentiment": 2, "topic": 3},

            # 使用体验 - 负面
            {"text": "操作复杂，学习曲线陡峭", "sentiment": 0, "topic": 3},
            {"text": "体验不佳，有很多不便之处", "sentiment": 0, "topic": 3},

            # 中性评论
            {"text": "产品还可以，没什么特别的感觉", "sentiment": 1, "topic": 0},
            {"text": "服务一般，没有特别好也没有特别差", "sentiment": 1, "topic": 1},
            {"text": "价格适中，不算贵也不算便宜", "sentiment": 1, "topic": 2},
            {"text": "用起来还行，没什么大问题", "sentiment": 1, "topic": 3},
        ]

        # 复制数据以增加规模
        enlarged_data = []
        for item in extended_data:
            for i in range(15):
                new_item = item.copy()
                enlarged_data.append(new_item)

        return enlarged_data

    def create_balanced_dataset(self):
        """创建平衡的数据集"""
        datasets = self._create_extended_dataset()

        # 添加特殊否定模式样本
        special_samples = [
            {"text": "这个产品不太差，还算可以", "sentiment": 2, "topic": 0},
            {"text": "质量不算差，对得起价格", "sentiment": 2, "topic": 0},
            {"text": "服务不糟糕，基本满意", "sentiment": 2, "topic": 1},
            {"text": "价格不贵，性价比不错", "sentiment": 2, "topic": 2},
            {"text": "用起来不难用，操作简单", "sentiment": 2, "topic": 3},
        ]

        datasets.extend(special_samples * 5)

        # 统计分布
        sentiment_counts = {}
        topic_counts = {}
        for item in datasets:
            sentiment = item['sentiment']
            topic = item['topic']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        print(f"情感分布: {sentiment_counts}")
        print(f"主题分布: {topic_counts}")

        return datasets

    def train_model(self, model, train_loader, val_loader, num_epochs=3, model_type='sentiment'):
        """训练单个模型"""
        model.to(self.device)

        # 优化器
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        # 学习率调度器
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = 0
        patience = 3
        patience_counter = 0

        # 初始化训练历史记录
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        # 发送训练开始消息
        if self.training_callback:
            self.training_callback({
                'type': 'training_start',
                'model_type': model_type,
                'total_epochs': num_epochs
            })

        print(f"\n开始训练{model_type}模型...")

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # 更新进度条
                current_loss = loss.item()
                current_acc = 100 * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

                # 发送批次更新消息
                if self.training_callback and batch_idx % 10 == 0:  # 每10个批次更新一次
                    self.training_callback({
                        'type': 'batch_update',
                        'model_type': model_type,
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'total_batches': len(train_loader),
                        'loss': current_loss,
                        'accuracy': current_acc
                    })

            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = total_loss / len(train_loader)

            # 保存训练历史
            train_loss_history.append(avg_train_loss)
            train_acc_history.append(train_accuracy)

            # 验证阶段
            val_accuracy, avg_val_loss = self.evaluate_model(model, val_loader, criterion)
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(val_accuracy)

            # 发送epoch完成消息
            if self.training_callback:
                self.training_callback({
                    'type': 'epoch_complete',
                    'model_type': model_type,
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_acc': train_accuracy,
                    'val_loss': avg_val_loss,
                    'val_acc': val_accuracy
                })

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
            print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')

            # 早停
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"早停在第 {epoch + 1} 轮")
                break

        # 发送训练完成消息
        if self.training_callback:
            self.training_callback({
                'type': 'training_complete',
                'model_type': model_type,
                'train_loss_history': train_loss_history,
                'train_acc_history': train_acc_history,
                'val_loss_history': val_loss_history,
                'val_acc_history': val_acc_history
            })

        return model

    def evaluate_model(self, model, val_loader, criterion):
        """评估模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)

        return accuracy, avg_loss

    def train_models(self, num_epochs=3):
        """训练情感分析和主题分类模型"""
        print("正在准备训练数据...")

        # 发送数据准备消息
        if self.training_callback:
            self.training_callback({
                'type': 'data_preparation',
                'message': '正在准备训练数据...'
            })

        datasets = self.create_balanced_dataset()

        # 提取文本和标签
        texts = [item['text'] for item in datasets]
        sentiment_labels = [item['sentiment'] for item in datasets]
        topic_labels = [item['topic'] for item in datasets]

        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]

        # 划分训练集和验证集
        (train_texts, val_texts,
         train_sentiment, val_sentiment,
         train_topic, val_topic) = train_test_split(
            processed_texts, sentiment_labels, topic_labels,
            test_size=0.2, random_state=42, stratify=sentiment_labels
        )

        # 创建数据加载器
        sentiment_train_dataset = TextDataset(train_texts, train_sentiment, self.tokenizer, self.max_length)
        sentiment_val_dataset = TextDataset(val_texts, val_sentiment, self.tokenizer, self.max_length)
        topic_train_dataset = TextDataset(train_texts, train_topic, self.tokenizer, self.max_length)
        topic_val_dataset = TextDataset(val_texts, val_topic, self.tokenizer, self.max_length)

        sentiment_train_loader = DataLoader(sentiment_train_dataset, batch_size=self.batch_size, shuffle=True)
        sentiment_val_loader = DataLoader(sentiment_val_dataset, batch_size=self.batch_size)
        topic_train_loader = DataLoader(topic_train_dataset, batch_size=self.batch_size, shuffle=True)
        topic_val_loader = DataLoader(topic_val_dataset, batch_size=self.batch_size)

        # 初始化模型
        self.sentiment_model = SentimentBERTModel(num_classes=3)
        self.topic_model = TopicBERTModel(num_classes=4)

        # 训练情感分析模型
        if self.training_callback:
            self.training_callback({
                'type': 'model_training_start',
                'model_type': '情感分析',
                'message': '开始训练情感分析模型...'
            })

        self.sentiment_model = self.train_model(
            self.sentiment_model, sentiment_train_loader, sentiment_val_loader,
            num_epochs, 'sentiment'
        )

        # 训练主题分类模型
        if self.training_callback:
            self.training_callback({
                'type': 'model_training_start',
                'model_type': '主题分类',
                'message': '开始训练主题分类模型...'
            })

        self.topic_model = self.train_model(
            self.topic_model, topic_train_loader, topic_val_loader,
            num_epochs, 'topic'
        )

        # 保存模型
        self.save_models()

        # 最终评估
        print("\n=== 最终模型评估 ===")
        sentiment_accuracy, _ = self.evaluate_model(
            self.sentiment_model, sentiment_val_loader, nn.CrossEntropyLoss()
        )
        topic_accuracy, _ = self.evaluate_model(
            self.topic_model, topic_val_loader, nn.CrossEntropyLoss()
        )

        print(f"情感分析验证准确率: {sentiment_accuracy:.2f}%")
        print(f"主题分类验证准确率: {topic_accuracy:.2f}%")

        # 发送训练完成消息
        if self.training_callback:
            self.training_callback({
                'type': 'training_final_results',
                'sentiment_accuracy': sentiment_accuracy,
                'topic_accuracy': topic_accuracy,
                'message': '模型训练完成！'
            })

        return True

    def _analyze_negation_context(self, text):
        """分析否定上下文"""
        if not isinstance(text, str):
            return None

        words = list(jieba.cut(text))

        # 检查特殊否定模式
        for pattern, sentiment in self.special_negation_patterns.items():
            if pattern in text:
                return sentiment, f"特殊否定模式: {pattern}"

        # 动态分析否定+负面词组合
        for i, word in enumerate(words):
            if word in self.negation_words and i < len(words) - 1:
                next_word = words[i + 1]
                if next_word in self.negative_boost_words:
                    if i > 0 and words[i - 1] in ['还', '也', '都', '挺', '算']:
                        return 2, f"否定+负面词组合: {words[i - 1]}{word}{next_word}"
                    else:
                        return 2, f"否定+负面词组合: {word}{next_word}"

        return None

    def _apply_sentiment_rules(self, text, base_sentiment, base_confidence):
        """应用情感判断规则进行修正"""
        if not isinstance(text, str):
            return base_sentiment, base_confidence

        # 检查特殊否定模式
        negation_result = self._analyze_negation_context(text)
        if negation_result is not None:
            sentiment, reason = negation_result
            return sentiment, max(base_confidence, 80.0)

        text_lower = text

        # 强负面词规则
        for word in self.strong_negative_words:
            if word in text_lower:
                words = list(jieba.cut(text_lower))
                word_index = text_lower.find(word)
                if word_index > 0:
                    preceding_text = text_lower[:word_index]
                    preceding_words = list(jieba.cut(preceding_text))
                    if any(neg_word in preceding_words for neg_word in self.negation_words):
                        continue
                return 0, max(base_confidence, 85.0)

        # 强正面词规则
        for word in self.strong_positive_words:
            if word in text_lower:
                return 2, max(base_confidence, 85.0)

        return base_sentiment, base_confidence

    def predict_sentiment(self, text):
        """预测情感"""
        if self.sentiment_model is None:
            raise Exception("情感分析模型未加载")

        self.sentiment_model.eval()
        processed_text = self.preprocess_text(text)

        if not processed_text.strip():
            return 1, 0.0  # 中性，0置信度

        encoding = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            outputs = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            sentiment = predicted.item()
            confidence_score = confidence.item() * 100

            # 规则修正
            corrected_sentiment, corrected_confidence = self._apply_sentiment_rules(
                text, sentiment, confidence_score
            )

            return corrected_sentiment, corrected_confidence

    def predict_topic(self, text):
        """预测主题"""
        if self.topic_model is None:
            raise Exception("主题分类模型未加载")

        self.topic_model.eval()
        processed_text = self.preprocess_text(text)

        if not processed_text.strip():
            return 0, 0.0  # 默认主题，0置信度

        encoding = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            outputs = self.topic_model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            return predicted.item(), confidence.item() * 100

    def analyze_text(self, text):
        """分析单条文本"""
        sentiment, sentiment_conf = self.predict_sentiment(text)
        topic, topic_conf = self.predict_topic(text)

        result = {
            'text': text,
            'sentiment': self.sentiment_labels[sentiment],
            'sentiment_confidence': round(sentiment_conf, 2),
            'topic': self.topic_labels[topic],
            'topic_confidence': round(topic_conf, 2)
        }

        return result

    def batch_analyze(self, texts):
        """批量分析文本"""
        results = []
        for text in texts:
            try:
                result = self.analyze_text(text)
                results.append(result)
            except Exception as e:
                print(f"分析文本失败: {text}, 错误: {e}")
        return results

    def visualize_results(self, texts, output_file='result.png'):
        """可视化分析结果并保存到文件"""
        results = self.batch_analyze(texts)

        if not results:
            print("没有有效的结果可可视化")
            return None

        # 情感分布
        sentiments = [result['sentiment'] for result in results]
        sentiment_counts = pd.Series(sentiments).value_counts()

        # 主题分布
        topics = [result['topic'] for result in results]
        topic_counts = pd.Series(topics).value_counts()

        plt.figure(figsize=(12, 5))

        # 情感分布饼图
        plt.subplot(1, 2, 1)
        if len(sentiment_counts) > 0:
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
            plt.title('情感分布')
        else:
            plt.title('无情感数据')

        # 主题分布条形图
        plt.subplot(1, 2, 2)
        if len(topic_counts) > 0:
            plt.bar(topic_counts.index, topic_counts.values)
            plt.xticks(rotation=45)
            plt.title('主题分布')
        else:
            plt.title('无主题数据')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file


class TrainingProgressWindow:
    """训练进度窗口"""

    def __init__(self, parent, analyzer):
        self.parent = parent
        self.analyzer = analyzer
        self.window = tk.Toplevel(parent)
        self.window.title("模型训练进度")
        self.window.geometry("800x600")

        # 设置回调函数
        self.analyzer.set_training_callback(self.update_progress)

        # 消息队列
        self.message_queue = queue.Queue()

        # 进度变量
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_model = ""

        self.setup_ui()

        # 开始检查消息队列
        self.check_queue()

    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="模型训练进度监控",
                                font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # 当前状态
        self.status_var = tk.StringVar(value="等待训练开始...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                 font=('Arial', 11))
        status_label.grid(row=1, column=0, pady=(0, 10), sticky=tk.W)

        # 进度条框架
        progress_frame = ttk.LabelFrame(main_frame, text="训练进度", padding="10")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)

        # 当前模型标签
        self.model_var = tk.StringVar(value="当前模型: -")
        model_label = ttk.Label(progress_frame, textvariable=self.model_var,
                                font=('Arial', 10))
        model_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        # 当前epoch标签
        self.epoch_var = tk.StringVar(value="当前轮次: - / -")
        epoch_label = ttk.Label(progress_frame, textvariable=self.epoch_var,
                                font=('Arial', 10))
        epoch_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

        # 总体进度条
        self.overall_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.overall_progress.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # epoch进度条
        self.epoch_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.epoch_progress.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 指标框架
        metrics_frame = ttk.LabelFrame(main_frame, text="训练指标", padding="10")
        metrics_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 创建两列的网格
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)

        # 训练损失
        self.train_loss_var = tk.StringVar(value="训练损失: -")
        train_loss_label = ttk.Label(metrics_frame, textvariable=self.train_loss_var)
        train_loss_label.grid(row=0, column=0, sticky=tk.W, pady=2)

        # 验证损失
        self.val_loss_var = tk.StringVar(value="验证损失: -")
        val_loss_label = ttk.Label(metrics_frame, textvariable=self.val_loss_var)
        val_loss_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        # 训练准确率
        self.train_acc_var = tk.StringVar(value="训练准确率: -")
        train_acc_label = ttk.Label(metrics_frame, textvariable=self.train_acc_var)
        train_acc_label.grid(row=1, column=0, sticky=tk.W, pady=2)

        # 验证准确率
        self.val_acc_var = tk.StringVar(value="验证准确率: -")
        val_acc_label = ttk.Label(metrics_frame, textvariable=self.val_acc_var)
        val_acc_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        # 详细日志
        log_frame = ttk.LabelFrame(main_frame, text="训练日志", padding="10")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10,
                                                  font=('Microsoft YaHei', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, pady=(10, 0))

        self.close_button = ttk.Button(button_frame, text="关闭窗口",
                                       command=self.close_window, state='disabled')
        self.close_button.grid(row=0, column=0, padx=5)

    def update_progress(self, data):
        """更新训练进度（从训练线程调用）"""
        self.message_queue.put(data)

    def check_queue(self):
        """检查消息队列并更新UI"""
        try:
            while True:
                data = self.message_queue.get_nowait()
                self.process_message(data)
        except queue.Empty:
            pass

        # 继续检查
        self.window.after(100, self.check_queue)

    def process_message(self, data):
        """处理消息"""
        msg_type = data.get('type', '')

        if msg_type == 'data_preparation':
            self.status_var.set("准备训练数据...")
            self.log_message(data.get('message', ''))

        elif msg_type == 'model_training_start':
            self.current_model = data.get('model_type', '')
            self.model_var.set(f"当前模型: {self.current_model}")
            self.status_var.set(f"开始训练{self.current_model}模型...")
            self.log_message(data.get('message', ''))

        elif msg_type == 'training_start':
            self.total_epochs = data.get('total_epochs', 5)
            self.epoch_var.set(f"当前轮次: 0 / {self.total_epochs}")

        elif msg_type == 'batch_update':
            epoch = data.get('epoch', 1)
            batch = data.get('batch', 1)
            total_batches = data.get('total_batches', 1)

            # 更新epoch进度
            batch_progress = (batch / total_batches) * 100
            self.epoch_progress['value'] = batch_progress

            # 更新总体进度（假设有两个模型，每个有total_epochs轮）
            model_index = 0 if self.current_model == "情感分析" else 1
            overall_progress = ((model_index * self.total_epochs + (epoch - 1)) /
                                (2 * self.total_epochs)) * 100 + (batch_progress / (2 * self.total_epochs))
            self.overall_progress['value'] = overall_progress

            self.status_var.set(f"训练{self.current_model}模型 - Epoch {epoch} - Batch {batch}/{total_batches}")

        elif msg_type == 'epoch_complete':
            epoch = data.get('epoch', 1)
            self.current_epoch = epoch
            self.epoch_var.set(f"当前轮次: {epoch} / {self.total_epochs}")

            # 更新指标
            train_loss = data.get('train_loss', 0)
            train_acc = data.get('train_acc', 0)
            val_loss = data.get('val_loss', 0)
            val_acc = data.get('val_acc', 0)

            self.train_loss_var.set(f"训练损失: {train_loss:.4f}")
            self.train_acc_var.set(f"训练准确率: {train_acc:.2f}%")
            self.val_loss_var.set(f"验证损失: {val_loss:.4f}")
            self.val_acc_var.set(f"验证准确率: {val_acc:.2f}%")

            self.log_message(f"Epoch {epoch}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, "
                             f"验证损失={val_loss:.4f}, 验证准确率={val_acc:.2f}%")

        elif msg_type == 'training_complete':
            self.log_message(f"{self.current_model}模型训练完成")

        elif msg_type == 'training_final_results':
            sentiment_acc = data.get('sentiment_accuracy', 0)
            topic_acc = data.get('topic_accuracy', 0)

            self.status_var.set("模型训练完成！")
            self.log_message(f"训练完成！情感分析准确率: {sentiment_acc:.2f}%，主题分类准确率: {topic_acc:.2f}%")
            self.log_message(data.get('message', ''))

            # 训练完成，启用关闭按钮
            self.close_button['state'] = 'normal'

    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def close_window(self):
        """关闭窗口"""
        self.window.destroy()


class BERTAnalyzerGUI:
    def __init__(self):
        self.analyzer = None
        self.root = tk.Tk()
        self.root.title("BERT智能文本情感分析与主题分类系统")
        self.root.geometry("900x700")

        # 设置图标（如果有的话）
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass

        self.setup_gui()

    def setup_gui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="BERT智能文本情感分析与主题分类系统",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # 模型状态显示
        self.model_status_var = tk.StringVar(value="模型状态: 未加载")
        status_label = ttk.Label(main_frame, textvariable=self.model_status_var,
                                 font=('Arial', 10))
        status_label.grid(row=1, column=0, columnspan=3, pady=5)

        # 控制按钮框架
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        # 训练模型按钮
        train_btn = ttk.Button(control_frame, text="训练新模型", command=self.train_model)
        train_btn.grid(row=0, column=0, padx=5)

        # 加载模型按钮
        load_btn = ttk.Button(control_frame, text="加载现有模型", command=self.load_model)
        load_btn.grid(row=0, column=1, padx=5)

        # 测试模型按钮
        test_btn = ttk.Button(control_frame, text="测试模型", command=self.test_model)
        test_btn.grid(row=0, column=2, padx=5)

        # 输入文本区域
        input_label = ttk.Label(main_frame, text="输入文本:", font=('Arial', 10, 'bold'))
        input_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 5))

        self.input_text = scrolledtext.ScrolledText(main_frame, width=60, height=10,
                                                    font=('Microsoft YaHei', 10))
        self.input_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # 添加示例文本按钮
        example_btn = ttk.Button(main_frame, text="插入示例文本", command=self.insert_example)
        example_btn.grid(row=5, column=0, pady=5)

        # 清空文本按钮
        clear_btn = ttk.Button(main_frame, text="清空文本", command=self.clear_text)
        clear_btn.grid(row=5, column=1, pady=5)

        # 分析按钮
        analyze_btn = ttk.Button(main_frame, text="分析文本", command=self.analyze_text,
                                 style='Accent.TButton')
        analyze_btn.grid(row=5, column=2, pady=5)

        # 结果显示区域
        result_label = ttk.Label(main_frame, text="分析结果:", font=('Arial', 10, 'bold'))
        result_label.grid(row=6, column=0, sticky=tk.W, pady=(20, 5))

        self.result_text = scrolledtext.ScrolledText(main_frame, width=60, height=15,
                                                     font=('Microsoft YaHei', 10), state='disabled')
        self.result_text.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # 配置样式
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

        # 初始化分析器
        self.initialize_analyzer()

    def initialize_analyzer(self):
        """初始化BERT分析器"""
        try:
            self.analyzer = BERTsentimentAnalyzer(
                max_length=128,
                batch_size=8,
                learning_rate=2e-5
            )

            # 检查模型是否存在
            if self.analyzer.check_models_exist():
                if messagebox.askyesno("模型检测", "检测到本地已存在训练好的模型文件，是否加载？"):
                    self.load_model()
                else:
                    self.model_status_var.set("模型状态: 未加载 (本地有模型文件)")
            else:
                self.model_status_var.set("模型状态: 未加载")

        except Exception as e:
            messagebox.showerror("错误", f"初始化失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        try:
            self.status_var.set("正在加载模型...")
            self.root.update()

            success = self.analyzer.load_models()
            if success:
                self.model_status_var.set("模型状态: 已加载 (情感分析 + 主题分类)")
                messagebox.showinfo("成功", "模型加载成功！")
            else:
                messagebox.showwarning("警告", "模型加载失败，请训练新模型")

        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
        finally:
            self.status_var.set("就绪")

    def train_model(self):
        """训练模型"""
        if not messagebox.askyesno("确认", "训练新模型将覆盖现有模型，是否继续？"):
            return

        # 创建进度窗口
        self.progress_window = TrainingProgressWindow(self.root, self.analyzer)

        def train_thread():
            try:
                start_time = time.time()
                success = self.analyzer.train_models(num_epochs=5)
                end_time = time.time()

                if success:
                    elapsed = end_time - start_time
                    # 进度窗口会显示完成消息，这里不需要再弹窗
                    self.model_status_var.set("模型状态: 已加载 (情感分析 + 主题分类)")
                    # 重新加载训练好的模型
                    self.load_model()
                else:
                    messagebox.showerror("错误", "模型训练失败")

            except Exception as e:
                messagebox.showerror("错误", f"训练失败: {str(e)}")

        # 在新线程中训练
        Thread(target=train_thread, daemon=True).start()

    def test_model(self):
        """测试模型"""
        if self.analyzer is None or self.analyzer.sentiment_model is None:
            messagebox.showwarning("警告", "请先加载或训练模型")
            return

        test_texts = [
            "这个产品真的很不错，质量很好，价格也合理",
            "客服态度很差，解决问题效率低下",
            "物流速度很快，包装也很精美",
            "功能一般，没有特别突出的地方",
            "价格太贵了，性价比不高",
            "使用体验很棒，操作很简单",
            "垃圾产品，千万别买",
            "服务态度差到极点",
            "这个东西不好用，非常失望",
            "非常满意，物超所值，强烈推荐",
            "这个产品不太差，还算可以接受",
            "质量不算差，对得起这个价格",
            "用起来不难用，基本功能都有",
        ]

        try:
            self.status_var.set("正在测试模型...")
            self.root.update()

            results = self.analyzer.batch_analyze(test_texts)

            # 显示结果
            self.result_text.config(state='normal')
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== BERT模型测试结果 ===\n\n")

            for i, result in enumerate(results, 1):
                self.result_text.insert(tk.END, f"{i}. 文本: {result['text']}\n")
                self.result_text.insert(tk.END,
                                        f"   情感: {result['sentiment']} (置信度: {result['sentiment_confidence']}%)\n")
                self.result_text.insert(tk.END,
                                        f"   主题: {result['topic']} (置信度: {result['topic_confidence']}%)\n\n")

            self.result_text.config(state='disabled')

            # 可视化
            output_file = self.analyzer.visualize_results(test_texts, 'test_result.png')
            if output_file:
                messagebox.showinfo("测试完成", f"测试完成！结果已保存到 {output_file}")

        except Exception as e:
            messagebox.showerror("错误", f"测试失败: {str(e)}")
        finally:
            self.status_var.set("就绪")

    def analyze_text(self):
        """分析文本"""
        if self.analyzer is None or self.analyzer.sentiment_model is None:
            messagebox.showwarning("警告", "请先加载或训练模型")
            return

        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入要分析的文本")
            return

        try:
            self.status_var.set("正在分析文本...")
            self.root.update()

            # 处理多行文本
            texts = [line.strip() for line in text.split('\n') if line.strip()]
            results = self.analyzer.batch_analyze(texts)

            # 显示结果
            self.result_text.config(state='normal')
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== 分析结果 ===\n\n")

            for i, result in enumerate(results, 1):
                self.result_text.insert(tk.END, f"{i}. 文本: {result['text']}\n")
                self.result_text.insert(tk.END,
                                        f"   情感: {result['sentiment']} (置信度: {result['sentiment_confidence']}%)\n")
                self.result_text.insert(tk.END,
                                        f"   主题: {result['topic']} (置信度: {result['topic_confidence']}%)\n\n")

            self.result_text.config(state='disabled')

            # 如果有多条文本，保存可视化结果
            if len(texts) > 1:
                output_file = self.analyzer.visualize_results(texts, 'analysis_result.png')
                if output_file:
                    messagebox.showinfo("分析完成", f"分析完成！可视化结果已保存到 {output_file}")
            else:
                messagebox.showinfo("分析完成", "分析完成！")

        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")
        finally:
            self.status_var.set("就绪")

    def insert_example(self):
        """插入示例文本"""
        examples = [
            "这个产品质量非常好，性价比超高！",
            "客服响应太慢了，等了半天没人理",
            "物流速度不错，包装也很用心",
            "价格有点贵，但是质量确实好",
            "使用起来很顺手，界面很友好"
        ]

        self.input_text.delete(1.0, tk.END)
        for example in examples:
            self.input_text.insert(tk.END, example + "\n\n")

    def clear_text(self):
        """清空文本"""
        self.input_text.delete(1.0, tk.END)

    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    print("=== BERT优化版智能文本情感分析与主题分类系统 ===\n")
    import jieba
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
    from sklearn.model_selection import train_test_split
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import re
    import warnings
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from tqdm import tqdm
    import time
    import tkinter as tk
    from tkinter import scrolledtext, messagebox, filedialog, ttk
    from threading import Thread
    import queue
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    warnings.filterwarnings('ignore')

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if os.path.exists("C:\\Users\\DELL\\Desktop\\simhei.ttf"):
            plt.rcParams['font.sans-serif'] = ['SimHei']
            print("已设置SimHei字体")
        else:
            print("使用默认字体")
    except Exception as e:
        print(f"字体设置失败: {e}")

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-chinese',
                cache_dir='./models',
                use_auth_token=False,
                proxies=None
            )
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    class SentimentBERTModel(nn.Module):
        """情感分析BERT模型"""

        def __init__(self, num_classes=3, dropout_rate=0.3):
            super(SentimentBERTModel, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            output = self.dropout(pooled_output)
            logits = self.classifier(output)
            return logits

    class TopicBERTModel(nn.Module):
        """主题分类BERT模型"""

        def __init__(self, num_classes=4, dropout_rate=0.3):
            super(TopicBERTModel, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            output = self.dropout(pooled_output)
            logits = self.classifier(output)
            return logits

    class BERTsentimentAnalyzer:
        def __init__(self, max_length=128, batch_size=16, learning_rate=2e-5):
            self.max_length = max_length
            self.batch_size = batch_size
            self.learning_rate = learning_rate

            # 初始化BERT分词器
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

            # 初始化模型
            self.sentiment_model = None
            self.topic_model = None

            # 模型文件路径
            self.sentiment_model_path = 'best_sentiment_model.pth'
            self.topic_model_path = 'best_topic_model.pth'

            # 标签映射
            self.sentiment_labels = {0: '负面', 1: '中性', 2: '正面'}
            self.topic_labels = {0: '产品评价', 1: '客户服务', 2: '价格反馈', 3: '使用体验'}

            # 设备配置
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {self.device}")

            # 情感关键词（用于规则修正）
            self.negative_boost_words = {
                '差', '坏', '垃圾', '失望', '讨厌', '不好', '糟糕', '坑', '贵', '慢',
                '难用', '卡顿', '故障', '问题', '缺陷', '不足', '差劲', '垃圾', '骗人',
                '坑爹', '垃圾货', '劣质', '破烂', '废物', '没用', '浪费时间', '后悔',
                '别买', '不推荐', '上当', '被骗', '垃圾产品', '质量差', '服务差', '很差',
                '非常差', '极差', '太差', '最差', '不好用', '难用', '垃圾服务'
            }

            self.positive_boost_words = {
                '好', '棒', '满意', '喜欢', '不错', '优秀', '推荐', '超值', '舒服',
                '完美', '惊喜', '赞', '漂亮', '流畅', '便捷', '快速', '专业', '值得',
                '物超所值', '性价比高', '很好', '非常好', '特别棒', '强烈推荐', '很棒',
                '非常好', '极其满意', '非常满意', '特别好', '非常不错'
            }

            self.strong_negative_words = {'垃圾', '骗人', '上当', '别买', '不推荐', '后悔', '差劲', '劣质', '坑爹',
                                          '废物'}
            self.strong_positive_words = {'完美', '惊喜', '强烈推荐', '物超所值', '特别棒', '极其满意', '非常满意'}
            self.negation_words = {'不', '没', '无', '未', '别', '莫', '勿', '休', '免', '非'}

            self.special_negation_patterns = {
                '不太差': 2, '不算差': 2, '不算坏': 2, '不糟糕': 2, '不差': 2,
                '不坏': 2, '不难用': 2, '不慢': 2, '不贵': 2, '不太贵': 2, '不算贵': 2,
            }
            # 训练进度回调函数
            self.training_callback = None

        def set_training_callback(self, callback):
            """设置训练进度回调函数"""
            self.training_callback = callback

        def check_models_exist(self):
            """检查模型文件是否存在"""
            return os.path.exists(self.sentiment_model_path) and os.path.exists(self.topic_model_path)

        def load_models(self):
            try:
                # 初始化模型结构
                self.sentiment_model = SentimentBERTModel(num_classes=3)
                self.topic_model = TopicBERTModel(num_classes=4)

                # 加载模型权重
                self.sentiment_model.load_state_dict(torch.load(self.sentiment_model_path, map_location=self.device))
                self.topic_model.load_state_dict(torch.load(self.topic_model_path, map_location=self.device))

                # 将模型移动到设备
                self.sentiment_model.to(self.device)
                self.topic_model.to(self.device)

                print("模型加载成功!")
                return True
            except Exception as e:
                print(f"模型加载失败: {e}")
                return False

        def save_models(self):
            """保存模型到文件"""
            try:
                if self.sentiment_model is not None:
                    torch.save(self.sentiment_model.state_dict(), self.sentiment_model_path)
                    print(f"情感模型已保存到: {self.sentiment_model_path}")

                if self.topic_model is not None:
                    torch.save(self.topic_model.state_dict(), self.topic_model_path)
                    print(f"主题模型已保存到: {self.topic_model_path}")

                return True
            except Exception as e:
                print(f"模型保存失败: {e}")
                return False

        def preprocess_text(self, text):
            """文本预处理"""
            if not isinstance(text, str):
                return ""
            # 去除特殊字符和数字，保留中文和基本标点
            text = re.sub(r'[^\u4e00-\u9fa5，。！？]', ' ', text)
            return text.strip()

        def _create_extended_dataset(self):
            """创建扩展的示例数据集"""
            extended_data = [
                # 产品评价 - 正面
                {"text": "这个产品质量很好，做工精细，用料扎实", "sentiment": 2, "topic": 0},
                {"text": "外观设计漂亮，颜色搭配很时尚", "sentiment": 2, "topic": 0},
                {"text": "功能齐全，操作简单易懂", "sentiment": 2, "topic": 0},
                {"text": "性能稳定，运行流畅不卡顿", "sentiment": 2, "topic": 0},

                # 产品评价 - 负面
                {"text": "质量一般，细节处理不够好", "sentiment": 0, "topic": 0},
                {"text": "设计普通，没有什么特色", "sentiment": 0, "topic": 0},
                {"text": "功能有限，不能满足需求", "sentiment": 0, "topic": 0},
                {"text": "性能不稳定，经常出现问题", "sentiment": 0, "topic": 0},

                # 客户服务 - 正面
                {"text": "客服态度很好，回答问题耐心", "sentiment": 2, "topic": 1},
                {"text": "售后服务到位，问题解决及时", "sentiment": 2, "topic": 1},
                {"text": "技术支持专业，解决方案有效", "sentiment": 2, "topic": 1},

                # 客户服务 - 负面
                {"text": "客服响应慢，等待时间太长", "sentiment": 0, "topic": 1},
                {"text": "售后服务差，推卸责任", "sentiment": 0, "topic": 1},
                {"text": "技术不专业，问题反复出现", "sentiment": 0, "topic": 1},

                # 价格反馈 - 正面
                {"text": "价格实惠，性价比很高", "sentiment": 2, "topic": 2},
                {"text": "物超所值，质量对得起价格", "sentiment": 2, "topic": 2},
                {"text": "促销活动很给力，价格优惠", "sentiment": 2, "topic": 2},

                # 价格反馈 - 负面
                {"text": "价格偏贵，性价比不高", "sentiment": 0, "topic": 2},
                {"text": "定价不合理，同价位有更好选择", "sentiment": 0, "topic": 2},

                # 使用体验 - 正面
                {"text": "使用方便，上手很快", "sentiment": 2, "topic": 3},
                {"text": "体验流畅，感觉很舒适", "sentiment": 2, "topic": 3},
                {"text": "操作简单，学习成本低", "sentiment": 2, "topic": 3},

                # 使用体验 - 负面
                {"text": "操作复杂，学习曲线陡峭", "sentiment": 0, "topic": 3},
                {"text": "体验不佳，有很多不便之处", "sentiment": 0, "topic": 3},

                # 中性评论
                {"text": "产品还可以，没什么特别的感觉", "sentiment": 1, "topic": 0},
                {"text": "服务一般，没有特别好也没有特别差", "sentiment": 1, "topic": 1},
                {"text": "价格适中，不算贵也不算便宜", "sentiment": 1, "topic": 2},
                {"text": "用起来还行，没什么大问题", "sentiment": 1, "topic": 3},
            ]

            # 复制数据以增加规模
            enlarged_data = []
            for item in extended_data:
                for i in range(15):
                    new_item = item.copy()
                    enlarged_data.append(new_item)

            return enlarged_data

        def create_balanced_dataset(self):
            """创建平衡的数据集"""
            datasets = self._create_extended_dataset()

            # 添加特殊否定模式样本
            special_samples = [
                {"text": "这个产品不太差，还算可以", "sentiment": 2, "topic": 0},
                {"text": "质量不算差，对得起价格", "sentiment": 2, "topic": 0},
                {"text": "服务不糟糕，基本满意", "sentiment": 2, "topic": 1},
                {"text": "价格不贵，性价比不错", "sentiment": 2, "topic": 2},
                {"text": "用起来不难用，操作简单", "sentiment": 2, "topic": 3},
            ]

            datasets.extend(special_samples * 5)

            # 统计分布
            sentiment_counts = {}
            topic_counts = {}
            for item in datasets:
                sentiment = item['sentiment']
                topic = item['topic']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            print(f"情感分布: {sentiment_counts}")
            print(f"主题分布: {topic_counts}")

            return datasets

        def train_model(self, model, train_loader, val_loader, num_epochs=3, model_type='sentiment'):
            """训练单个模型"""
            model.to(self.device)

            # 优化器
            optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)

            # 学习率调度器
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )

            # 损失函数
            criterion = nn.CrossEntropyLoss()

            best_val_accuracy = 0
            patience = 3
            patience_counter = 0

            # 初始化训练历史记录
            train_loss_history = []
            train_acc_history = []
            val_loss_history = []
            val_acc_history = []

            # 发送训练开始消息
            if self.training_callback:
                self.training_callback({
                    'type': 'training_start',
                    'model_type': model_type,
                    'total_epochs': num_epochs
                })

            print(f"\n开始训练{model_type}模型...")

            for epoch in range(num_epochs):
                # 训练阶段
                model.train()
                total_loss = 0
                train_correct = 0
                train_total = 0

                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
                for batch_idx, batch in enumerate(train_pbar):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    # 更新进度条
                    current_loss = loss.item()
                    current_acc = 100 * train_correct / train_total
                    train_pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Acc': f'{current_acc:.2f}%'
                    })

                    # 发送批次更新消息
                    if self.training_callback and batch_idx % 10 == 0:  # 每10个批次更新一次
                        self.training_callback({
                            'type': 'batch_update',
                            'model_type': model_type,
                            'epoch': epoch + 1,
                            'batch': batch_idx + 1,
                            'total_batches': len(train_loader),
                            'loss': current_loss,
                            'accuracy': current_acc
                        })

                train_accuracy = 100 * train_correct / train_total
                avg_train_loss = total_loss / len(train_loader)

                # 保存训练历史
                train_loss_history.append(avg_train_loss)
                train_acc_history.append(train_accuracy)

                # 验证阶段
                val_accuracy, avg_val_loss = self.evaluate_model(model, val_loader, criterion)
                val_loss_history.append(avg_val_loss)
                val_acc_history.append(val_accuracy)

                # 发送epoch完成消息
                if self.training_callback:
                    self.training_callback({
                        'type': 'epoch_complete',
                        'model_type': model_type,
                        'epoch': epoch + 1,
                        'train_loss': avg_train_loss,
                        'train_acc': train_accuracy,
                        'val_loss': avg_val_loss,
                        'val_acc': val_accuracy
                    })

                print(f'Epoch {epoch + 1}/{num_epochs}:')
                print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
                print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')

                # 早停
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"早停在第 {epoch + 1} 轮")
                    break

            # 发送训练完成消息
            if self.training_callback:
                self.training_callback({
                    'type': 'training_complete',
                    'model_type': model_type,
                    'train_loss_history': train_loss_history,
                    'train_acc_history': train_acc_history,
                    'val_loss_history': val_loss_history,
                    'val_acc_history': val_acc_history
                })

            return model

        def evaluate_model(self, model, val_loader, criterion):
            """评估模型"""
            model.eval()
            total_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            avg_loss = total_loss / len(val_loader)

            return accuracy, avg_loss

        def train_models(self, num_epochs=5):
            """训练情感分析和主题分类模型"""
            print("正在准备训练数据...")

            # 发送数据准备消息
            if self.training_callback:
                self.training_callback({
                    'type': 'data_preparation',
                    'message': '正在准备训练数据...'
                })

            datasets = self.create_balanced_dataset()

            # 提取文本和标签
            texts = [item['text'] for item in datasets]
            sentiment_labels = [item['sentiment'] for item in datasets]
            topic_labels = [item['topic'] for item in datasets]

            # 预处理文本
            processed_texts = [self.preprocess_text(text) for text in texts]

            # 划分训练集和验证集
            (train_texts, val_texts,
             train_sentiment, val_sentiment,
             train_topic, val_topic) = train_test_split(
                processed_texts, sentiment_labels, topic_labels,
                test_size=0.2, random_state=42, stratify=sentiment_labels
            )

            # 创建数据加载器
            sentiment_train_dataset = TextDataset(train_texts, train_sentiment, self.tokenizer, self.max_length)
            sentiment_val_dataset = TextDataset(val_texts, val_sentiment, self.tokenizer, self.max_length)
            topic_train_dataset = TextDataset(train_texts, train_topic, self.tokenizer, self.max_length)
            topic_val_dataset = TextDataset(val_texts, val_topic, self.tokenizer, self.max_length)

            sentiment_train_loader = DataLoader(sentiment_train_dataset, batch_size=self.batch_size, shuffle=True)
            sentiment_val_loader = DataLoader(sentiment_val_dataset, batch_size=self.batch_size)
            topic_train_loader = DataLoader(topic_train_dataset, batch_size=self.batch_size, shuffle=True)
            topic_val_loader = DataLoader(topic_val_dataset, batch_size=self.batch_size)

            # 初始化模型
            self.sentiment_model = SentimentBERTModel(num_classes=3)
            self.topic_model = TopicBERTModel(num_classes=4)

            # 训练情感分析模型
            if self.training_callback:
                self.training_callback({
                    'type': 'model_training_start',
                    'model_type': '情感分析',
                    'message': '开始训练情感分析模型...'
                })

            self.sentiment_model = self.train_model(
                self.sentiment_model, sentiment_train_loader, sentiment_val_loader,
                num_epochs, 'sentiment'
            )

            # 训练主题分类模型
            if self.training_callback:
                self.training_callback({
                    'type': 'model_training_start',
                    'model_type': '主题分类',
                    'message': '开始训练主题分类模型...'
                })

            self.topic_model = self.train_model(
                self.topic_model, topic_train_loader, topic_val_loader,
                num_epochs, 'topic'
            )

            # 保存模型
            self.save_models()

            # 最终评估
            print("\n=== 最终模型评估 ===")
            sentiment_accuracy, _ = self.evaluate_model(
                self.sentiment_model, sentiment_val_loader, nn.CrossEntropyLoss()
            )
            topic_accuracy, _ = self.evaluate_model(
                self.topic_model, topic_val_loader, nn.CrossEntropyLoss()
            )

            print(f"情感分析验证准确率: {sentiment_accuracy:.2f}%")
            print(f"主题分类验证准确率: {topic_accuracy:.2f}%")

            # 发送训练完成消息
            if self.training_callback:
                self.training_callback({
                    'type': 'training_final_results',
                    'sentiment_accuracy': sentiment_accuracy,
                    'topic_accuracy': topic_accuracy,
                    'message': '模型训练完成！'
                })

            return True

        def _analyze_negation_context(self, text):
            """分析否定上下文"""
            if not isinstance(text, str):
                return None

            words = list(jieba.cut(text))

            # 检查特殊否定模式
            for pattern, sentiment in self.special_negation_patterns.items():
                if pattern in text:
                    return sentiment, f"特殊否定模式: {pattern}"

            # 动态分析否定+负面词组合
            for i, word in enumerate(words):
                if word in self.negation_words and i < len(words) - 1:
                    next_word = words[i + 1]
                    if next_word in self.negative_boost_words:
                        if i > 0 and words[i - 1] in ['还', '也', '都', '挺', '算']:
                            return 2, f"否定+负面词组合: {words[i - 1]}{word}{next_word}"
                        else:
                            return 2, f"否定+负面词组合: {word}{next_word}"

            return None

        def _apply_sentiment_rules(self, text, base_sentiment, base_confidence):
            """应用情感判断规则进行修正"""
            if not isinstance(text, str):
                return base_sentiment, base_confidence

            # 检查特殊否定模式
            negation_result = self._analyze_negation_context(text)
            if negation_result is not None:
                sentiment, reason = negation_result
                return sentiment, max(base_confidence, 80.0)

            text_lower = text

            # 强负面词规则
            for word in self.strong_negative_words:
                if word in text_lower:
                    words = list(jieba.cut(text_lower))
                    word_index = text_lower.find(word)
                    if word_index > 0:
                        preceding_text = text_lower[:word_index]
                        preceding_words = list(jieba.cut(preceding_text))
                        if any(neg_word in preceding_words for neg_word in self.negation_words):
                            continue
                    return 0, max(base_confidence, 85.0)

            # 强正面词规则
            for word in self.strong_positive_words:
                if word in text_lower:
                    return 2, max(base_confidence, 85.0)

            return base_sentiment, base_confidence

        def predict_sentiment(self, text):
            """预测情感"""
            if self.sentiment_model is None:
                raise Exception("情感分析模型未加载")

            self.sentiment_model.eval()
            processed_text = self.preprocess_text(text)

            if not processed_text.strip():
                return 1, 0.0  # 中性，0置信度

            encoding = self.tokenizer(
                processed_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                sentiment = predicted.item()
                confidence_score = confidence.item() * 100

                # 规则修正
                corrected_sentiment, corrected_confidence = self._apply_sentiment_rules(
                    text, sentiment, confidence_score
                )

                return corrected_sentiment, corrected_confidence

        def predict_topic(self, text):
            """预测主题"""
            if self.topic_model is None:
                raise Exception("主题分类模型未加载")

            self.topic_model.eval()
            processed_text = self.preprocess_text(text)

            if not processed_text.strip():
                return 0, 0.0  # 默认主题，0置信度

            encoding = self.tokenizer(
                processed_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.topic_model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                return predicted.item(), confidence.item() * 100

        def analyze_text(self, text):
            """分析单条文本"""
            sentiment, sentiment_conf = self.predict_sentiment(text)
            topic, topic_conf = self.predict_topic(text)

            result = {
                'text': text,
                'sentiment': self.sentiment_labels[sentiment],
                'sentiment_confidence': round(sentiment_conf, 2),
                'topic': self.topic_labels[topic],
                'topic_confidence': round(topic_conf, 2)
            }

            return result

        def batch_analyze(self, texts):
            """批量分析文本"""
            results = []
            for text in texts:
                try:
                    result = self.analyze_text(text)
                    results.append(result)
                except Exception as e:
                    print(f"分析文本失败: {text}, 错误: {e}")
            return results

        def visualize_results(self, texts, output_file='result.png'):
            """可视化分析结果并保存到文件"""
            results = self.batch_analyze(texts)

            if not results:
                print("没有有效的结果可可视化")
                return None

            # 情感分布
            sentiments = [result['sentiment'] for result in results]
            sentiment_counts = pd.Series(sentiments).value_counts()

            # 主题分布
            topics = [result['topic'] for result in results]
            topic_counts = pd.Series(topics).value_counts()

            plt.figure(figsize=(12, 5))

            # 情感分布饼图
            plt.subplot(1, 2, 1)
            if len(sentiment_counts) > 0:
                plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
                plt.title('情感分布')
            else:
                plt.title('无情感数据')

            # 主题分布条形图
            plt.subplot(1, 2, 2)
            if len(topic_counts) > 0:
                plt.bar(topic_counts.index, topic_counts.values)
                plt.xticks(rotation=45)
                plt.title('主题分布')
            else:
                plt.title('无主题数据')

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            return output_file

    class TrainingProgressWindow:
        """训练进度窗口"""

        def __init__(self, parent, analyzer):
            self.parent = parent
            self.analyzer = analyzer
            self.window = tk.Toplevel(parent)
            self.window.title("模型训练进度")
            self.window.geometry("800x600")

            # 设置回调函数
            self.analyzer.set_training_callback(self.update_progress)

            # 消息队列
            self.message_queue = queue.Queue()

            # 进度变量
            self.current_epoch = 0
            self.total_epochs = 0
            self.current_model = ""

            self.setup_ui()

            # 开始检查消息队列
            self.check_queue()

        def setup_ui(self):
            """设置UI界面"""
            # 主框架
            main_frame = ttk.Frame(self.window, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # 配置网格权重
            self.window.columnconfigure(0, weight=1)
            self.window.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(2, weight=1)

            # 标题
            title_label = ttk.Label(main_frame, text="模型训练进度监控",
                                    font=('Arial', 14, 'bold'))
            title_label.grid(row=0, column=0, pady=(0, 10))

            # 当前状态
            self.status_var = tk.StringVar(value="等待训练开始...")
            status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                     font=('Arial', 11))
            status_label.grid(row=1, column=0, pady=(0, 10), sticky=tk.W)

            # 进度条框架
            progress_frame = ttk.LabelFrame(main_frame, text="训练进度", padding="10")
            progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            progress_frame.columnconfigure(0, weight=1)

            # 当前模型标签
            self.model_var = tk.StringVar(value="当前模型: -")
            model_label = ttk.Label(progress_frame, textvariable=self.model_var,
                                    font=('Arial', 10))
            model_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

            # 当前epoch标签
            self.epoch_var = tk.StringVar(value="当前轮次: - / -")
            epoch_label = ttk.Label(progress_frame, textvariable=self.epoch_var,
                                    font=('Arial', 10))
            epoch_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

            # 总体进度条
            self.overall_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
            self.overall_progress.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

            # epoch进度条
            self.epoch_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
            self.epoch_progress.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

            # 指标框架
            metrics_frame = ttk.LabelFrame(main_frame, text="训练指标", padding="10")
            metrics_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

            # 创建两列的网格
            metrics_frame.columnconfigure(0, weight=1)
            metrics_frame.columnconfigure(1, weight=1)

            # 训练损失
            self.train_loss_var = tk.StringVar(value="训练损失: -")
            train_loss_label = ttk.Label(metrics_frame, textvariable=self.train_loss_var)
            train_loss_label.grid(row=0, column=0, sticky=tk.W, pady=2)

            # 验证损失
            self.val_loss_var = tk.StringVar(value="验证损失: -")
            val_loss_label = ttk.Label(metrics_frame, textvariable=self.val_loss_var)
            val_loss_label.grid(row=0, column=1, sticky=tk.W, pady=2)

            # 训练准确率
            self.train_acc_var = tk.StringVar(value="训练准确率: -")
            train_acc_label = ttk.Label(metrics_frame, textvariable=self.train_acc_var)
            train_acc_label.grid(row=1, column=0, sticky=tk.W, pady=2)

            # 验证准确率
            self.val_acc_var = tk.StringVar(value="验证准确率: -")
            val_acc_label = ttk.Label(metrics_frame, textvariable=self.val_acc_var)
            val_acc_label.grid(row=1, column=1, sticky=tk.W, pady=2)

            # 详细日志
            log_frame = ttk.LabelFrame(main_frame, text="训练日志", padding="10")
            log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(0, weight=1)

            self.log_text = scrolledtext.ScrolledText(log_frame, height=10,
                                                      font=('Microsoft YaHei', 9))
            self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # 控制按钮
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=5, column=0, pady=(10, 0))

            self.close_button = ttk.Button(button_frame, text="关闭窗口",
                                           command=self.close_window, state='disabled')
            self.close_button.grid(row=0, column=0, padx=5)

        def update_progress(self, data):
            """更新训练进度（从训练线程调用）"""
            self.message_queue.put(data)

        def check_queue(self):
            """检查消息队列并更新UI"""
            try:
                while True:
                    data = self.message_queue.get_nowait()
                    self.process_message(data)
            except queue.Empty:
                pass

            # 继续检查
            self.window.after(100, self.check_queue)

        def process_message(self, data):
            """处理消息"""
            msg_type = data.get('type', '')

            if msg_type == 'data_preparation':
                self.status_var.set("准备训练数据...")
                self.log_message(data.get('message', ''))

            elif msg_type == 'model_training_start':
                self.current_model = data.get('model_type', '')
                self.model_var.set(f"当前模型: {self.current_model}")
                self.status_var.set(f"开始训练{self.current_model}模型...")
                self.log_message(data.get('message', ''))

            elif msg_type == 'training_start':
                self.total_epochs = data.get('total_epochs', 5)
                self.epoch_var.set(f"当前轮次: 0 / {self.total_epochs}")

            elif msg_type == 'batch_update':
                epoch = data.get('epoch', 1)
                batch = data.get('batch', 1)
                total_batches = data.get('total_batches', 1)

                # 更新epoch进度
                batch_progress = (batch / total_batches) * 100
                self.epoch_progress['value'] = batch_progress

                # 更新总体进度（假设有两个模型，每个有total_epochs轮）
                model_index = 0 if self.current_model == "情感分析" else 1
                overall_progress = ((model_index * self.total_epochs + (epoch - 1)) /
                                    (2 * self.total_epochs)) * 100 + (batch_progress / (2 * self.total_epochs))
                self.overall_progress['value'] = overall_progress

                self.status_var.set(f"训练{self.current_model}模型 - Epoch {epoch} - Batch {batch}/{total_batches}")

            elif msg_type == 'epoch_complete':
                epoch = data.get('epoch', 1)
                self.current_epoch = epoch
                self.epoch_var.set(f"当前轮次: {epoch} / {self.total_epochs}")

                # 更新指标
                train_loss = data.get('train_loss', 0)
                train_acc = data.get('train_acc', 0)
                val_loss = data.get('val_loss', 0)
                val_acc = data.get('val_acc', 0)

                self.train_loss_var.set(f"训练损失: {train_loss:.4f}")
                self.train_acc_var.set(f"训练准确率: {train_acc:.2f}%")
                self.val_loss_var.set(f"验证损失: {val_loss:.4f}")
                self.val_acc_var.set(f"验证准确率: {val_acc:.2f}%")

                self.log_message(f"Epoch {epoch}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, "
                                 f"验证损失={val_loss:.4f}, 验证准确率={val_acc:.2f}%")

            elif msg_type == 'training_complete':
                self.log_message(f"{self.current_model}模型训练完成")

            elif msg_type == 'training_final_results':
                sentiment_acc = data.get('sentiment_accuracy', 0)
                topic_acc = data.get('topic_accuracy', 0)

                self.status_var.set("模型训练完成！")
                self.log_message(f"训练完成！情感分析准确率: {sentiment_acc:.2f}%，主题分类准确率: {topic_acc:.2f}%")
                self.log_message(data.get('message', ''))

                # 训练完成，启用关闭按钮
                self.close_button['state'] = 'normal'

        def log_message(self, message):
            """添加日志消息"""
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)

        def close_window(self):
            """关闭窗口"""
            self.window.destroy()

    class BERTAnalyzerGUI:
        def __init__(self):
            self.analyzer = None
            self.root = tk.Tk()
            self.root.title("BERT智能文本情感分析与主题分类系统")
            self.root.geometry("900x700")

            # 设置图标（如果有的话）
            try:
                self.root.iconbitmap('icon.ico')
            except:
                pass

            self.setup_gui()

        def setup_gui(self):
            # 创建主框架
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # 配置网格权重
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(3, weight=1)

            # 标题
            title_label = ttk.Label(main_frame, text="BERT智能文本情感分析与主题分类系统",
                                    font=('Arial', 16, 'bold'))
            title_label.grid(row=0, column=0, columnspan=3, pady=10)

            # 模型状态显示
            self.model_status_var = tk.StringVar(value="模型状态: 未加载")
            status_label = ttk.Label(main_frame, textvariable=self.model_status_var,
                                     font=('Arial', 10))
            status_label.grid(row=1, column=0, columnspan=3, pady=5)

            # 控制按钮框架
            control_frame = ttk.Frame(main_frame)
            control_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

            # 训练模型按钮
            train_btn = ttk.Button(control_frame, text="训练新模型", command=self.train_model)
            train_btn.grid(row=0, column=0, padx=5)

            # 加载模型按钮
            load_btn = ttk.Button(control_frame, text="加载现有模型", command=self.load_model)
            load_btn.grid(row=0, column=1, padx=5)

            # 测试模型按钮
            test_btn = ttk.Button(control_frame, text="测试模型", command=self.test_model)
            test_btn.grid(row=0, column=2, padx=5)

            # 输入文本区域
            input_label = ttk.Label(main_frame, text="输入文本:", font=('Arial', 10, 'bold'))
            input_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 5))

            self.input_text = scrolledtext.ScrolledText(main_frame, width=60, height=10,
                                                        font=('Microsoft YaHei', 10))
            self.input_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

            # 添加示例文本按钮
            example_btn = ttk.Button(main_frame, text="插入示例文本", command=self.insert_example)
            example_btn.grid(row=5, column=0, pady=5)

            # 清空文本按钮
            clear_btn = ttk.Button(main_frame, text="清空文本", command=self.clear_text)
            clear_btn.grid(row=5, column=1, pady=5)

            # 分析按钮
            analyze_btn = ttk.Button(main_frame, text="分析文本", command=self.analyze_text,
                                     style='Accent.TButton')
            analyze_btn.grid(row=5, column=2, pady=5)

            # 结果显示区域
            result_label = ttk.Label(main_frame, text="分析结果:", font=('Arial', 10, 'bold'))
            result_label.grid(row=6, column=0, sticky=tk.W, pady=(20, 5))

            self.result_text = scrolledtext.ScrolledText(main_frame, width=60, height=15,
                                                         font=('Microsoft YaHei', 10), state='disabled')
            self.result_text.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

            # 底部状态栏
            self.status_var = tk.StringVar(value="就绪")
            status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
            status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

            # 配置样式
            style = ttk.Style()
            style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

            # 初始化分析器
            self.initialize_analyzer()

        def initialize_analyzer(self):
            """初始化BERT分析器"""
            try:
                self.analyzer = BERTsentimentAnalyzer(
                    max_length=128,
                    batch_size=8,
                    learning_rate=2e-5
                )

                # 检查模型是否存在
                if self.analyzer.check_models_exist():
                    if messagebox.askyesno("模型检测", "检测到本地已存在训练好的模型文件，是否加载？"):
                        self.load_model()
                    else:
                        self.model_status_var.set("模型状态: 未加载 (本地有模型文件)")
                else:
                    self.model_status_var.set("模型状态: 未加载")

            except Exception as e:
                messagebox.showerror("错误", f"初始化失败: {str(e)}")

        def load_model(self):
            """加载模型"""
            try:
                self.status_var.set("正在加载模型...")
                self.root.update()

                success = self.analyzer.load_models()
                if success:
                    self.model_status_var.set("模型状态: 已加载 (情感分析 + 主题分类)")
                    messagebox.showinfo("成功", "模型加载成功！")
                else:
                    messagebox.showwarning("警告", "模型加载失败，请训练新模型")

            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            finally:
                self.status_var.set("就绪")

        def train_model(self):
            """训练模型"""
            if not messagebox.askyesno("确认", "训练新模型将覆盖现有模型，是否继续？"):
                return

            # 创建进度窗口
            self.progress_window = TrainingProgressWindow(self.root, self.analyzer)

            def train_thread():
                try:
                    start_time = time.time()
                    success = self.analyzer.train_models(num_epochs=5)
                    end_time = time.time()

                    if success:
                        elapsed = end_time - start_time
                        # 进度窗口会显示完成消息，这里不需要再弹窗
                        self.model_status_var.set("模型状态: 已加载 (情感分析 + 主题分类)")
                        # 重新加载训练好的模型
                        self.load_model()
                    else:
                        messagebox.showerror("错误", "模型训练失败")

                except Exception as e:
                    messagebox.showerror("错误", f"训练失败: {str(e)}")

            # 在新线程中训练
            Thread(target=train_thread, daemon=True).start()

        def test_model(self):
            """测试模型"""
            if self.analyzer is None or self.analyzer.sentiment_model is None:
                messagebox.showwarning("警告", "请先加载或训练模型")
                return

            test_texts = [
                "这个产品真的很不错，质量很好，价格也合理",
                "客服态度很差，解决问题效率低下",
                "物流速度很快，包装也很精美",
                "功能一般，没有特别突出的地方",
                "价格太贵了，性价比不高",
                "使用体验很棒，操作很简单",
                "垃圾产品，千万别买",
                "服务态度差到极点",
                "这个东西不好用，非常失望",
                "非常满意，物超所值，强烈推荐",
                "这个产品不太差，还算可以接受",
                "质量不算差，对得起这个价格",
                "用起来不难用，基本功能都有",
            ]

            try:
                self.status_var.set("正在测试模型...")
                self.root.update()

                results = self.analyzer.batch_analyze(test_texts)

                # 显示结果
                self.result_text.config(state='normal')
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "=== BERT模型测试结果 ===\n\n")

                for i, result in enumerate(results, 1):
                    self.result_text.insert(tk.END, f"{i}. 文本: {result['text']}\n")
                    self.result_text.insert(tk.END,
                                            f"   情感: {result['sentiment']} (置信度: {result['sentiment_confidence']}%)\n")
                    self.result_text.insert(tk.END,
                                            f"   主题: {result['topic']} (置信度: {result['topic_confidence']}%)\n\n")

                self.result_text.config(state='disabled')

                # 可视化
                output_file = self.analyzer.visualize_results(test_texts, 'test_result.png')
                if output_file:
                    messagebox.showinfo("测试完成", f"测试完成！结果已保存到 {output_file}")

            except Exception as e:
                messagebox.showerror("错误", f"测试失败: {str(e)}")
            finally:
                self.status_var.set("就绪")

        def analyze_text(self):
            """分析文本"""
            if self.analyzer is None or self.analyzer.sentiment_model is None:
                messagebox.showwarning("警告", "请先加载或训练模型")
                return

            text = self.input_text.get(1.0, tk.END).strip()
            if not text:
                messagebox.showwarning("警告", "请输入要分析的文本")
                return

            try:
                self.status_var.set("正在分析文本...")
                self.root.update()

                # 处理多行文本
                texts = [line.strip() for line in text.split('\n') if line.strip()]
                results = self.analyzer.batch_analyze(texts)

                # 显示结果
                self.result_text.config(state='normal')
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "=== 分析结果 ===\n\n")

                for i, result in enumerate(results, 1):
                    self.result_text.insert(tk.END, f"{i}. 文本: {result['text']}\n")
                    self.result_text.insert(tk.END,
                                            f"   情感: {result['sentiment']} (置信度: {result['sentiment_confidence']}%)\n")
                    self.result_text.insert(tk.END,
                                            f"   主题: {result['topic']} (置信度: {result['topic_confidence']}%)\n\n")

                self.result_text.config(state='disabled')

                # 如果有多条文本，保存可视化结果
                if len(texts) > 1:
                    output_file = self.analyzer.visualize_results(texts, 'analysis_result.png')
                    if output_file:
                        messagebox.showinfo("分析完成", f"分析完成！可视化结果已保存到 {output_file}")
                else:
                    messagebox.showinfo("分析完成", "分析完成！")

            except Exception as e:
                messagebox.showerror("错误", f"分析失败: {str(e)}")
            finally:
                self.status_var.set("就绪")

        def insert_example(self):
            """插入示例文本"""
            examples = [
                "这个产品质量非常好，性价比超高！",
                "客服响应太慢了，等了半天没人理",
                "物流速度不错，包装也很用心",
                "价格有点贵，但是质量确实好",
                "使用起来很顺手，界面很友好"
            ]

            self.input_text.delete(1.0, tk.END)
            for example in examples:
                self.input_text.insert(tk.END, example + "\n\n")

        def clear_text(self):
            """清空文本"""
            self.input_text.delete(1.0, tk.END)

        def run(self):
            """运行GUI"""
            self.root.mainloop()

def main():
    """主函数"""
    print("=== BERT优化版智能文本情感分析与主题分类系统 ===\n")
    print("重要提示：BERT模型需要从Hugging Face下载，由于网络限制，推荐使用VPN以使用全部功能")

if __name__ == "__main__":
        main()
    # 直接启动GUI模式
try:
    app = BERTAnalyzerGUI()
    app.run()
except Exception as e:
    print(f"GUI启动失败: {e}")
    print("程序退出")
