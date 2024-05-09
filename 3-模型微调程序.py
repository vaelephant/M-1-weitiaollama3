import datetime
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

#####################定义常量----开始################################

# 设置序列最大长度和数据类型
max_seq_length = 2048  # 选择任意长度！我们内部自动支持RoPE缩放！

dtype = None   # 自动检测数据类型。Tesla T4, V100用Float16，Ampere+用Bfloat16
load_in_4bit = True # 启用4bit量化以减少内存使用。可以设置为False

# 定义支持4位量化的模型列表
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    # 其他模型...
    "unsloth/llama-3-8b-bnb-4bit",
]


#####################定义常量----结束################################


#####################定义函数----开始################################
def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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


#####################定义函数----结束################################

# 第一步检查系统环境


print(f"{current_time()} - 环境检查如下：")
# 打印 PyTorch 的版本
print("PyTorch Version:", torch.__version__)

# 打印 CUDA 的版本，如果 CUDA 可用的话
print("CUDA Version:", torch.version.cuda)

# 检查 CUDA 是否可用
print("CUDA is available:", torch.cuda.is_available())

# 如果 CUDA 可用，打印出当前 CUDA 设备的数量和名称
if torch.cuda.is_available():
    print("Number of CUDA Devices:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))


# 第二步：加载模型（装货）

# 加载模型和分词器
print(f"{current_time()} - 正在加载模型和分词器...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print(f"{current_time()} - 模型加载完成。")



# 第三步：训练模式设置（装货）

# 模型优化设置
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 数据集的格式化函数
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(
            instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts, }


# 第四步：加载训练数据（给模型学习书籍）

# 加载和预处理数据集

file_path = "1-3-train-data/alpaca_gpt4_data_zh.json"
dataset = load_dataset("json", data_files={"train": file_path}, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)


# 第五步： 设置训练参数和训练模型
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
        max_steps=60,                     # 训练的最大轮数
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


# 第六步： 训练开始

start_time = datetime.datetime.now()
print(f"{current_time()} - 开始训练...")

# 训练前保存指标
pre_train_metrics = save_metrics(model)

# 开始训练
trainer_stats = trainer.train()


# # 第七步： 训练后保存指标

# post_train_metrics = save_metrics(model)

# # 比较训练前后的变化
# changes = compare_metrics(pre_train_metrics, post_train_metrics)

# # 打印变化情况
# print("Changes in parameters after training:")
# for name, change in changes.items():
#     print(f"{name}: {change}")

# # 使用不同的量化方法保存模型
# quantization_methods = ["q4_k_m", "q8_0", "f16"]
# for method in quantization_methods:
#     # 假设 save_pretrained_gguf 是一个扩展方法用于保存和量化模型
#     model.save_pretrained_gguf("dir", tokenizer, quantization_method=method)

#     print(f"{current_time()} - 保存训练好的模型完成。")



# 第八步：  训练结束时间间和持续时

end_time = datetime.datetime.now()
duration = end_time - start_time
print(f"{current_time()} - 训练完成。")
print(f"训练用时: {duration}")


# 保存模型到指定目录
model.save_pretrained_gguf("2-2-trained-model", tokenizer, quantization_method="q4_k_m")
model.save_pretrained_gguf("2-2-trained-model", tokenizer, quantization_method="q8_0")
model.save_pretrained_gguf("2-2-trained-model", tokenizer, quantization_method="f16")

print(f"{current_time()} - 保存训练好的模型完成。")
