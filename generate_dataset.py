import random
import pandas as pd

spam_words = ["win", "free", "money", "offer", "prize", "click", "urgent", "cash"]
ham_words = ["hello", "meeting", "project", "call", "assignment", "class", "update", "schedule"]

data = []

# 500 spam
for _ in range(500):
    msg = " ".join(random.choices(spam_words, k=5))
    data.append(["spam", msg])

# 500 ham
for _ in range(500):
    msg = " ".join(random.choices(ham_words, k=5))
    data.append(["ham", msg])

df = pd.DataFrame(data, columns=["label", "message"])
df.to_csv("spam.csv", index=False)

print("Dataset created")