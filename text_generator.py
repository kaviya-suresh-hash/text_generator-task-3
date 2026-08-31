from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125m"
)

prompt = "Artificial Intelligence is"

result = pipe(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.9
)

generated_text = result[0]["generated_text"]

print(generated_text)

with open("generated_text.txt", "w", encoding="utf-8") as file:
    file.write(generated_text)