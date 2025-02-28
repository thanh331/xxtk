import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Update the file path
file_path = '/content/drive/MyDrive/CNTT17-01_TranDucThanh_BKT2.xlsx' # Update 'MyDrive' to your folder if necessary
df = pd.read_excel(file_path, sheet_name="Sheet1")
print(df)
print(df.describe())
# Đổi tên cột cho đồng nhất
df.rename(columns={"Doanh thu": "Tổng giá trị (VND)", "Ngày": "Ngày Bán"}, inplace=True)

# Xử lý cột 'Ngày Bán'
df['Ngày Bán'].fillna(method='ffill', inplace=True)  # Điền bằng giá trị trước đó nếu thiếu
df['Ngày Bán'] = pd.to_datetime(df['Ngày Bán'], errors='coerce')  # Chuyển về datetime

# Chuyển đổi 'Tổng giá trị (VND)' sang kiểu số (loại bỏ dấu phẩy nếu có)
df["Tổng giá trị (VND)"] = df["Tổng giá trị (VND)"].astype(str).str.replace(",", "").astype(float)

# Loại bỏ giá trị NaN trong 'Ngày Bán' và 'Tổng giá trị (VND)'
df.dropna(subset=["Ngày Bán", "Tổng giá trị (VND)"], inplace=True)

# Tổng hợp doanh thu theo sản phẩm
product_sales = df.groupby("Tên sản phẩm")["Tổng giá trị (VND)"].sum()

# Vẽ biểu đồ tròn tỷ lệ doanh thu theo sản phẩm
if not product_sales.empty:
    plt.figure(figsize=(8, 8))
    plt.pie(product_sales, labels=product_sales.index, autopct="%1.1f%%", startangle=140,
            colors=plt.cm.Paired.colors, wedgeprops={"edgecolor": "black", "linewidth": 1})
    plt.title("Tỷ lệ doanh thu theo sản phẩm năm 2024", fontsize=14)
    plt.show()

# Biểu đồ phân phối tổng doanh thu
plt.figure(figsize=(10, 5))
sns.histplot(df["Tổng giá trị (VND)"], bins=20, kde=True, color="blue")
plt.xlabel("Tổng giá trị (VND)", fontsize=12)
plt.ylabel("Số lần xuất hiện", fontsize=12)
plt.title("Biểu đồ phân phối tổng doanh thu", fontsize=14)
plt.grid(True)
plt.show()

# Tổng hợp doanh thu theo ngày
daily_sales = df.groupby("Ngày Bán")["Tổng giá trị (VND)"].sum().reset_index()
daily_sales['Ngày Bán'] = (daily_sales['Ngày Bán'] - daily_sales['Ngày Bán'].min()).dt.days  # Chuyển ngày thành số ngày từ mốc đầu tiên

# Chia dữ liệu thành train/test
X = daily_sales[['Ngày Bán']]
y = daily_sales['Tổng giá trị (VND)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Biểu đồ xu hướng doanh thu theo thời gian
plt.figure(figsize=(12, 5))
plt.scatter(X_test, y_test, color='blue', label='Thực tế')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Dự đoán')
plt.xlabel("Số ngày từ ngày đầu tiên", fontsize=12)
plt.ylabel("Tổng doanh thu (VND)", fontsize=12)
plt.title("Xu hướng doanh thu theo thời gian (Thực tế vs Dự đoán)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Kết luận
print("Mô hình dự báo doanh thu hoạt động tốt với R2 Score:", r2)
print("Cải thiện mô hình bằng cách thử nghiệm các thuật toán khác hoặc thu thập thêm dữ liệu.")