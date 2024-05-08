import datetime
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 设置序列最大长度和数据类型
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 定义支持4位量化的模型列表
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    # 其他模型...
    "unsloth/llama-3-8b-bnb-4bit",
]

# 加载模型和分词器
print(f"{current_time()} - 正在加载模型和分词器...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print(f"{current_time()} - 模型加载完成。")

# 模型优化设置
# 获取或配置语言模型以使用 PEFT 方法进行高效微调
'''
使用 FastLanguageModel.get_peft_model 方法来获取或配置
一个特定的语言模型，应用了 PEFT（Parameter-Efficient Fine-Tuning）技术。
这段代码展示了如何为现有模型添加特定的微调参数，增强模型的微调能力而不显著增加参数数量。
PEFT、LoRA、LoFT-Q 等技术都是用于在保持预训练模型大部分参数不变的情况下进行有效的微调。
'''
model = FastLanguageModel.get_peft_model(
    model,                         # 原始的未微调的语言模型
    r=16,                          # 设置PEFT中降维的秩为16
    target_modules=[               # 指定要微调的模块名称列表
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,                 # LoRA扩展的维度，用于调整模型参数的广度
    lora_dropout=0,                # LoRA层中使用的dropout比率，这里设置为0表示不使用dropout
    bias="none",                   # 对PEFT模块中的偏置项设置，这里设置为'none'表示不使用偏置
    use_gradient_checkpointing="unsloth",  # 梯度检查点的使用策略，'unsloth'可能是特定策略的名称
    random_state=3407,             # 随机状态，用于确保初始化的一致性或重复实验的可复现性
    use_rslora=False,              # 是否使用可逆或改进的LoRA层
    loftq_config=None,             # 配置LoFT-Q，如果使用的话，这里为None表示不进行配置
)


# 数据集的格式化函数
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text": texts, }

# 加载和预处理数据集
file_path = "alpaca_gpt4_data_zh.json"
dataset = load_dataset("json", data_files={"train": file_path}, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 设置训练程序
trainer = SFTTrainer(
    model=model,                          # 要训练的模型
    tokenizer=tokenizer,                  # 使用的分词器
    train_dataset=dataset,                # 训练数据集
    dataset_text_field="text",            # 数据集中文本字段的名称
    max_seq_length=max_seq_length,        # 输入序列的最大长度
    dataset_num_proc=2,                   # 数据预处理时使用的进程数
    args=TrainingArguments(
        per_device_train_batch_size=2,    # 每个设备的训练批次大小
        gradient_accumulation_steps=4,    # 梯度累积步数，用于在内存限制下增大有效批次大小
        warmup_steps=5,                   # 预热步数，用于调整学习率
        max_steps=60,                     # 训练的最大步数
        learning_rate=2e-4,               # 初始学习率
        fp16=not torch.cuda.is_bf16_supported(),  # 如果设备不支持bf16，则使用fp16精度训练
        bf16=torch.cuda.is_bf16_supported(),      # 如果设备支持bf16，则使用bf16精度训练
        logging_steps=1,                  # 每隔多少步记录日志
        optim="adamw_8bit",               # 使用的优化器，这里是为8位计算优化的adamw
        weight_decay=0.01,                # 权重衰减，用于正则化和防止过拟合
        lr_scheduler_type="linear",       # 学习率调度器类型，这里使用线性调度
        seed=3407,                        # 随机种子，用于确保可复现性
        output_dir="outputs",             # 模型和日志的输出目录
    ),
)


# 保存模型训练前的权重或性能指标
def save_metrics(model):
    # 这里可以是权重的复制，也可以是其他性能指标
    return {name: param.clone() for name, param in model.named_parameters()}

# 比较模型训练前后的权重或性能指标
def compare_metrics(pre_metrics, post_metrics):
    changes = {}
    for name, pre_param in pre_metrics.items():
        post_param = post_metrics[name]
        changes[name] = torch.norm(post_param - pre_param)  # 计算差值的范数
    return changes



# 训练开始时间
start_time = datetime.datetime.now()
print(f"{current_time()} - 开始训练...")

# 训练前保存指标
pre_train_metrics = save_metrics(model)

# 开始训练
trainer_stats = trainer.train()

# 训练结束时间和持续时间
end_time = datetime.datetime.now()
duration = end_time - start_time
print(f"{current_time()} - 训练完成。")
print(f"训练用时: {duration}")


# 训练后保存指标
post_train_metrics = save_metrics(model)

# 比较训练前后的变化
changes = compare_metrics(pre_train_metrics, post_train_metrics)

# 打印变化情况
print("Changes in parameters after training:")
for name, change in changes.items():
    print(f"{name}: {change}")

# 使用不同的量化方法保存模型
quantization_methods = ["q4_k_m", "q8_0", "f16"]
for method in quantization_methods:
    # 假设 save_pretrained_gguf 是一个扩展方法用于保存和量化模型
    model.save_pretrained_gguf("dir", tokenizer, quantization_method=method)

    

# 保存模型到指定目录
model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m")
model.save_pretrained_gguf("dir", tokenizer, quantization_method="q8_0")
model.save_pretrained_gguf("dir", tokenizer, quantization_method="f16")
