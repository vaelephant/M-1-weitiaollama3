import datetime
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


# 第一步检查系统环境

print('环境检查如下：')
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


# 加载和预处理数据集
file_path = "alpaca_gpt4_data_zh.json"
dataset = load_dataset("json", data_files={"train": file_path}, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 设置训练参数和训练模型
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# 训练开始时间
start_time = datetime.datetime.now()
print(f"{current_time()} - 开始训练...")

# 开始训练
trainer_stats = trainer.train()

# 训练结束时间和持续时间
end_time = datetime.datetime.now()
duration = end_time - start_time
print(f"{current_time()} - 训练完成。")
print(f"训练用时: {duration}")

# 保存模型到指定目录
model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m")
model.save_pretrained_gguf("dir", tokenizer, quantization_method="q8_0")
model.save_pretrained_gguf("dir", tokenizer, quantization_method="f16")
