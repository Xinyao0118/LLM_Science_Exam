# LLM_Science_Exam
Use LLMs to answer difficult science questions

![competition](img/bgr.png)

[Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam)

## Competition Introduction
Inspired by the OpenBookQA dataset, this competition challenges participants to answer difficult science-based questions written by a Large Language Model.
- Type: NLP
- Recommended Model: BERT, LLAMA
- Evaluation Metrics:  Mean Average Precision @ 3
- Baseline: 

## Data Introduction

### **train.csv** - a set of 200 questions with the answer column \

One row example: \
__propmt__: What is the most popular explanation for the shower-curtain effect? \
__A__: The pressure differential between the inside and outside of the shower \
__B__: The decrease in velocity resulting in an increase in pressure \
__C__: The movement of air across the outside surface of the shower curtain \
__D__: The use of cold water \
__E__: Bernoulli's principle \
__answer__: E

### test.csv - the test set. Your task it to predict the top three most probable answers given the prompt. 
One row example: \
__propmt__: What is the term used in astrophysics to describe light-matter interactions resulting in energy shif... \
__A__: Blueshifting \
__B__: Redshifting \
__C__: Reddening \
__D__: Whitening \
__E__: Yellowing 

## Solution Approach

- Utilized the Pre-trained BERT model on Hugging face.
- Finetune the BERT model.
- Generate probabilities for each option with AutoModelForMultipleChoice class.
- Manipulate the output for 3 options with highest prob.
- Sturecture the pipeline for switching different models.

## Updates

- bert-base-cased model: CV 1.512489
- bert-base-multilingual-uncased model: CV 1.248739
- 

## Core code 

```Python
# --- skeleton Requirement: can generate different predictions for different models in Hugging Face.
class LLM_prediction:
    
    def __init__(self,model_path,options = 'ABCDE'):
        self.model_path = model_path
        self.options = options
        self.indices = list(range(len(options)))
        self.option_to_index = {option: index for option, index in zip(self.options, self.indices)}
        self.index_to_option = {index: option for option, index in zip(self.options, self.indices)}
        return
        
    def read_data(self,data_folder = None):
        #training
        train_df = pd.read_csv(f"{data_folder}/train.csv")
        self.train_ds = Dataset.from_pandas(train_df)
        #testing
        self.test_df = pd.read_csv(f"{data_folder}/test.csv")
        return self.train_ds,self.test_df
    
    def pre_process_data(self,row):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        question = [row['prompt']]*5
        answers = []
        for option in self.options:
            answers.append(row[option])
        tokenized_row = self.tokenizer(question,answers,truncation = True)
        tokenized_row['label'] = self.option_to_index[row['answer']]
        return tokenized_row
    
    def nlp(self,output_model_dir = 'finetuned_bert'):
        #return trainer
        model = AutoModelForMultipleChoice.from_pretrained(self.model_path)
        tokenized_train_ds = self.train_ds.map(self.pre_process_data, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
        training_args = TrainingArguments(
            output_dir=output_model_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            report_to='none'
        )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_train_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
        )
        self.trainer.train()
        return self.trainer
        
    def predictions_to_map_output(self,predictions):
        sorted_answer_indices = np.argsort(-predictions)
        top_answer_indices = sorted_answer_indices[:,:3] # Get the first three answers in each row
        top_answers = np.vectorize(self.index_to_option.get)(top_answer_indices)
        return np.apply_along_axis(lambda row: ' '.join(row), 1, top_answers)

    def inference(self,assign_random_answer = True):
        if not assign_random_answer:
            raise ValueError('Another inference way has not been be developed.')
        self.test_df['answer']='A'
        self.test_ds = Dataset.from_pandas(self.test_df)
        tokenized_test_ds = self.test_ds.map(self.pre_process_data, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
        predictions=self.trainer.predict(tokenized_test_ds)
        return self.predictions_to_map_output(predictions.predictions)
    
```

## Conclusion

