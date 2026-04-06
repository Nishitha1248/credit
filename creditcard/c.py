import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------



df = pd.read_csv("creditcard.csv")


# -----------------------------
# Step 2: Generate Card Numbers
# -------------------------------

def generate_card_number():
    prefixes = ['4', '51', '60']  # Visa, Mastercard, RuPay
    prefix = random.choice(prefixes)
    return prefix + ''.join([str(random.randint(0,9)) for _ in range(15)])


df['card_number'] = [generate_card_number() for _ in range(len(df))]

# -----------------------------
# Step 3: Detect Card Type
# -----------------------------
def get_card_type(card_number):
    card_number = str(card_number)

    if card_number.startswith('4'):
        return "Visa"

    if card_number.startswith(('51','52','53','54','55')):
        return "Mastercard"

    if card_number.startswith(('60','65','81','82')):
        return "RuPay"

    return "Unknown"

df['card_type'] = df['card_number'].apply(get_card_type)

# -----------------------------
# Step 4: Prepare Data
# -----------------------------
df.rename(columns={'Class': 'is_fraud'}, inplace=True)

# Use limited features (simple model)
X = df[['Time', 'Amount']]
y = df['is_fraud']

# Encode card_type
encoder = LabelEncoder()
df['card_type_encoded'] = encoder.fit_transform(df['card_type'])

# Add to features
X = df[['Time', 'Amount', 'card_type_encoded']]

# -----------------------------
# Step 5: Train Model
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Step 6: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# Step 7: User Input Prediction
# -----------------------------
print("\n--- Fraud Detection ---")

amount = float(input("Enter Transaction Amount: "))
time = float(input("Enter Transaction Time: "))
card_number = input("Enter Card Number: ")

# Detect card type
card_type = get_card_type(card_number)
print("Detected Card Type:", card_type)

# Encode manually
mapping = {"Visa": 0, "Mastercard": 1, "RuPay": 2, "Unknown": 3}
card_type_encoded = mapping.get(card_type, 3)

# Prediction
input_data = [[time, amount, card_type_encoded]]
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("⚠️ Fraud Transaction Detected!")
else:
    print("✅ Legitimate Transaction")