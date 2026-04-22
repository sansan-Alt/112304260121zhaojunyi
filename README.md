# 机器学习实验：基于 Word2Vec 的情感预测

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)

## 实验信息

- **实验名称**：Bag of Words Meets Bags of Popcorn
- **竞赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **学生姓名**：赵军毅
- **学生学号**：112304260121
- **实验日期**：2026-04-22

## 仓库结构

```
├── code/                  # 代码文件
│   ├── preprocessing.py   # 文本预处理模块
│   ├── bow_model.py       # Bag of Words 模型
│   ├── word2vec_model.py  # Word2Vec 模型
│   ├── main.py            # 主程序
│   └── requirements.txt   # 依赖包
├── report/                # 实验报告
│   ├── 实验报告.md         # 完整实验报告
│   └── readme_机器学习实验2模板.md # 实验模板
├── results/               # 实验结果
│   ├── submission_bow.csv      # Bag of Words 模型预测结果
│   └── submission_word2vec.csv # Word2Vec 模型预测结果
├── labeledTrainData.tsv/  # 训练集数据
├── testData.tsv/          # 测试集数据
├── unlabeledTrainData.tsv/# 无标签数据
├── .env.example           # 环境变量配置示例
├── .gitignore             # Git 忽略文件
└── README.md              # 项目说明文档
```

## 实验简介

本实验基于 Kaggle 竞赛 "Bag of Words Meets Bags of Popcorn"，实现了电影评论情感分析任务。使用了两种模型：

1. **Bag of Words (TF-IDF) 模型**
   
   - 交叉验证 AUC：0.9523
   - 特征：TF-IDF 向量化，5000 维特征

2. **Word2Vec 模型**
   
   - 训练集 AUC：0.9505
   - 特征：300 维 Word2Vec 词向量

## 环境要求

- Python 3.7+
- 依赖包：见 code/requirements.txt

## 使用方法

### 1. 克隆项目

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r code/requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，根据需要修改配置
```

### 5. 运行实验

```bash
python code/main.py
```

### 6. 提交结果

将 results/ 目录下的 CSV 文件提交到 Kaggle 竞赛页面。

## 配置说明

### 环境变量（.env 文件）

| 变量名             | 说明                   | 默认值    |
| --------------- | -------------------- | ------ |
| DATA_DIR        | 数据目录路径               | ./data |
| VECTORIZER_TYPE | 向量化器类型 (tfidf/count) | tfidf  |
| MAX_FEATURES    | 最大特征数                | 5000   |
| VECTOR_SIZE     | Word2Vec 向量维度        | 300    |
| EPOCHS          | 训练轮数                 | 10     |
| CV_FOLDS        | 交叉验证折数               | 5      |
| LOG_LEVEL       | 日志级别                 | INFO   |

## 实验结果

- **Bag of Words 模型**：AUC = 0.9431
- **Word2Vec 模型**：AUC = 0.9405

## 注意事项

- 数据文件较大，已按原始结构存放
- 模型文件（.pkl 和 .model）未上传到仓库，运行时会自动生成
- 每次实验后请及时提交代码和报告到 GitHub
- 建议使用虚拟环境管理依赖

## 许可证

本项目仅供学习和实验使用。
