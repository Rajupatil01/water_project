import pandas as pd
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv("smart_water_consumption_data.csv")

# MySQL credentials
user = "root"
password = "system1234"  # ← Replace this
host = "localhost"
database = "smart_water_monitoring"

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# Upload DataFrame to MySQL
df.to_sql(name='water_consumption', con=engine, if_exists='append', index=False)

print("✅ Data loaded successfully into MySQL!")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# ---------- DB Connection ----------
user = "root"
password = "system1234"  # ← Replace if needed
host = "localhost"
database = "smart_water_monitoring"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# ---------- Load Data ----------
df = pd.read_sql("SELECT * FROM water_consumption", con=engine)
df['date'] = pd.to_datetime(df['date'])

# ---------- Basic Info ----------
print("\n📊 Shape:", df.shape)
print("\n📋 Columns:", df.columns)
print("\n🧮 Stats:\n", df.describe())
print("\n❓ Missing Values:\n", df.isnull().sum())

# ---------- Daily Trend ----------
daily_trend = df.groupby('date')['daily_consumption_liters'].mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_trend, color='blue')
plt.title("📈 Average Daily Water Consumption")
plt.xlabel("Date")
plt.ylabel("Liters")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Leak Detection ----------
leak_df = df[df['daily_consumption_liters'] > 600]
print(f"\n💧 Possible Leaks Found: {leak_df.shape[0]}")
print(leak_df[['household_id', 'date', 'daily_consumption_liters']].head())

# ---------- Boxplot ----------
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['daily_consumption_liters'], color='red')
plt.title("📦 Outliers in Daily Consumption")
plt.tight_layout()
plt.show()
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- MySQL Connection ---
engine = create_engine("mysql+pymysql://root:system1234@localhost/smart_water_monitoring")

# --- Load Data ---
df = pd.read_sql("SELECT * FROM water_consumption", con=engine)

# --- Create Leak Label (1 = leak, 0 = normal) ---
df['leak'] = df['daily_consumption_liters'].apply(lambda x: 1 if x > 600 else 0)

# --- Feature Selection ---
features = ['daily_consumption_liters', 'pressure_psi', 'temperature_celsius', 'population_density']
X = df[features]
y = df['leak']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scaling (optional but useful) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train Classifier ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# --- Predict ---
y_pred = clf.predict(X_test)

# --- Accuracy & Metrics ---
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example input: [consumption, pressure, temp, population_density]
sample = [[720, 38.5, 28.2, 9600]]
sample_scaled = scaler.transform(sample)
leak_prediction = clf.predict(sample_scaled)
print("🔍 Leak Prediction:", "Leak Detected" if leak_prediction[0] == 1 else "No Leak")