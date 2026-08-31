from transformers import pipeline
import pandas as pd
import os

generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125m"
)

prompt = "Artificial Intelligence is"

temperatures = [0.2, 0.7, 1.0]
top_k_values = [20, 50, 100]

results = []

for temp in temperatures:
    for top_k in top_k_values:

        output = generator(
            prompt,
            max_new_tokens=50,
            temperature=temp,
            top_k=top_k,
            do_sample=True
        )

        generated_text = output[0]["generated_text"]

        results.append({
            "Prompt": prompt,
            "Temperature": temp,
            "Top_K": top_k,
            "Generated_Text": generated_text
        })

df = pd.DataFrame(results)

os.makedirs("dataset", exist_ok=True)

df.to_csv(
    "dataset/text_generation_results.csv",
    index=False
)

print("Parameter comparison results saved successfully!")