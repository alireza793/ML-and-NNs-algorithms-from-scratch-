import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv_sklearn(file_path, train_size=0.7, test_size=0.2, validation_size=0.1, random_state=None):
    """
    تقسیم یک فایل CSV به سه قسمت Train، Test و Validation با استفاده از scikit-learn.

    :param file_path: مسیر فایل CSV
    :param train_size: درصد داده‌ها برای مجموعه آموزش (پیش‌فرض: 70%)
    :param test_size: درصد داده‌ها برای مجموعه تست (پیش‌فرض: 20%)
    :param validation_size: درصد داده‌ها برای مجموعه اعتبارسنجی (پیش‌فرض: 10%)
    :param random_state: مقدار seed برای تکرارپذیری تقسیم‌بندی (پیش‌فرض: None)
    """

    # خواندن فایل CSV
    df = pd.read_csv(file_path)

    # جدا کردن ویژگی‌ها (X) و برچسب‌ها (y)
    # فرض می‌کنیم ستون آخر فایل CSV، برچسب‌ها هستند
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # تقسیم به Train و Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state)

    # تقسیم مجموعه Test به Test و Validation
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=validation_size / (test_size + validation_size), random_state=random_state)

    # ایجاد DataFrame برای هر مجموعه
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    validation_df = pd.concat([X_validation, y_validation], axis=1)

    # ذخیره هر مجموعه در یک فایل جداگانه
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    validation_df.to_csv('validation.csv', index=False)

    print("فایل CSV با موفقیت به سه قسمت تقسیم شد.")

# مثال استفاده:
file_path = 'your_file.csv'  # جایگزین با مسیر فایل CSV خود کنید
split_csv_sklearn(file_path, random_state=42)
'