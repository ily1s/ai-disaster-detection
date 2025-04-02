from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER")

text = "Flooding reported in Jakarta, Indonesia."
entities = ner(text)
print(entities)