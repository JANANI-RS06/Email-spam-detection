import tkinter as tk
from tkinter import messagebox
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load and train model (once)
# -------------------------------
data = pd.read_csv("spam_dataset.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# Tkinter UI
# -------------------------------
root = tk.Tk()
root.title("Email Spam Detection")
root.geometry("500x400")
root.resizable(False, False)

title = tk.Label(root, text="📧 Email Spam Detection", font=("Arial", 16, "bold"))
title.pack(pady=10)

accuracy_label = tk.Label(
    root, text=f"Model Accuracy: {accuracy*100:.2f}%", font=("Arial", 11)
)
accuracy_label.pack(pady=5)

input_label = tk.Label(root, text="Enter Email Content:", font=("Arial", 10))
input_label.pack(pady=5)

text_input = tk.Text(root, height=8, width=55)
text_input.pack(pady=5)

# -------------------------------
# Prediction function
# -------------------------------
def predict_spam():
    email_text = text_input.get("1.0", tk.END).strip()

    if not email_text:
        messagebox.showwarning("Input Error", "Please enter an email message.")
        return

    email_tfidf = tfidf.transform([email_text])
    prediction = model.predict(email_tfidf)[0]

    if prediction == 1:
        messagebox.showerror("Result", "🚨 This email is SPAM")
    else:
        messagebox.showinfo("Result", "✅ This email is NOT SPAM")

# -------------------------------
# Buttons
# -------------------------------
predict_btn = tk.Button(
    root, text="Predict", command=predict_spam, width=20, bg="#4CAF50", fg="white"
)
predict_btn.pack(pady=15)

def show_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Spam Detection")
    plt.show()

cm_btn = tk.Button(
    root, text="Show Confusion Matrix", command=show_confusion_matrix, width=20
)
cm_btn.pack(pady=5)

root.mainloop()
