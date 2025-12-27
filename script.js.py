
""""importing packages"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments


"""load pre trained model"""


model_name = "microsoft/resnet-50"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


"""importing data sets"""

dataset = load_dataset("imdb")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)
tokenized_dataset = dataset.map(preprocess_function, batched=True)




"""trainning"""

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=20,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
trainer.train()



"""evaluating"""

results = trainer.evaluate()
print(f"Validation Accuracy: {results['eval_accuracy']}")
