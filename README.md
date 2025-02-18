# Infini-Retri Package 

https://github.com/MrYxJ/InfiniRetri/ 

## How to use
Firstly, you can only using pip install package ```infini-retri``` to get our method.
```python
pip install infini-retri==0.0.3
```

### Our Method Initialization
It's very convenient. You just need to pass in the model and its tokenizer directly, or you can simply passing in the model name or path. Additionally, it should be noted that our method can only using in tranditional **attention-based** Transformer, and the parameter of "attn_implementation" currently only using **"eager"**.

```python  
from infini_retri import InfiniRetri

model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct" #  "./models/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation="eager") # attn_implementation only using "eager"
tokenizer = AutoTokenizer.from_pretrained(model_name)

ir = InfiniRetri(model, tokenizer)
# ir = InfiniRetri(name_or_path=model_name_or_path) 
```

### Our Method in Model Inference
Our methodis to design an innovative processing mechanisum during the ***model inference(fix model.generate())*** to handle texts that exceed the upper limit of the original context length. Specifically, when use, there are there parameters of inputting text: `context`, `question`, `prompt`, as follows:
- `context:` (str) required option in the input text, just the complete contextual content that needs to be procssed(no include question and prompt).
- `question`: (str) option parameter,  passed in is your question for the LLMs, for example, the Question text in QA Pair is passed in this parameter. For all tasks, this parameter is recommended to be filled in, as it has a significant impact on the understanding of the LLMs and the ability to provide correct answers.
- `prompt`:  (str) option parameter, the instruction template that concatenates the `context` and `question` text sections above, especially noting that **when concatenating the `context` and `question` text sections, two "\n\n" are used to distinguish the boundaries of the text in each section. This is a necessary condition for our method to run normally**.  For example, its default prompt template is `"Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion:\n\n{question}\n\nAnswer:"`.
"

In addition, three parameters are provided here for everyone to adjust according to different types of tasks to achieve the best answer effect, as follows:

- `window_length`: (int, default 1024) this controls the length of the context window during the execution of our method. When setting , it only need to ensure that it is less than the maximum context window of the your using model.
- `topk`: (int, default 300) It affects the cache capacity size during the operation of our method and the actual length of context to be processed throughout the inference process. In theory, the larger the value, the larger the retrieval range during the operation. The actual optimal value depends on the user's problem handling and can be self adjusted.
- `answer_length`:(int, default 8) It affects the effectiveness of outputting the correct answer, and its value can be set based on the user's expected token length of the correct answer in the `context` section. In theory, the closer the token length is set to the correct answer in the `context`, the better the effect of the model's answer under our method.


```python

# This short passage is extracted from HarryPotter just present usage of our menthod. 
# Due to its short length, it cannot demonstrate the advantages of our method in handling task on ultra long text. 
context = """
Harry woke at five o'clock the next 
morning and was too excited and nervous to go back to sleep. He got up and pulled on his jeans because he didn't want to walk into the station in his wizard's robes — he'd change on the train. He checked his Hogwarts list yet again to make sure he had everything he needed, saw that Hedwig was shut safely in her cage, and then paced the room, waiting for the Dursleys to get up. Two hours later, Harry's huge, heavy trunk had been loaded into the Dursleys’ car, Aunt Petunia had talked Dudley into sitting next to Harry, and they had set off.They reached King's Cross at half past ten. Uncle Vernon dumped Harry's trunk onto a cart and wheeled it into the station for him. Harry thought this was strangely kind until Uncle Vernon stopped dead, facing the platforms with a nasty grin on his face.
"""  

question = "Why did Harry decide to wear jeans instead of his wizard's robes to the train station?"

prompt = "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion:\n\n{question}\n\nAnswer:" # Note "\n\n" in boundary.

response = ir.generate(context=context, question=question, prompt=prompt)
print("Response:", response)
```
