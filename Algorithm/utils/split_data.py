import pandas as pd

def split_csv(file_path, train_size=0.7, test_size=0.2, validation_size=0.1):
    """
    تقسیم یک فایل CSV به سه قسمت Train، Test و Validation.

    :param file_path: مسیر فایل CSV
    :param train_size: درصد داده‌ها برای مجموعه آموزش (پیش‌فرض: 70%)
    :param test_size: درصد داده‌ها برای مجموعه تست (پیش‌فرض: 20%)
    :param validation_size: درصد داده‌ها برای مجموعه اعتبارسنجی (پیش‌فرض: 10%)
    """

    # خواندن فایل CSV
    df = pd.read_csv(file_path)

    # محاسبه اندازه هر مجموعه
    train_size_int = int(len(df) * train_size)
    test_size_int = int(len(df) * test_size)

    # تقسیم داده‌ها
    train_df = df[:train_size_int]
    test_df = df[train_size_int:train_size_int + test_size_int]
    validation_df = df[train_size_int + test_size_int:]

    # ذخیره هر مجموعه در یک فایل جداگانه
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    validation_df.to_csv('validation.csv', index=False)

    print("فایل CSV با موفقیت به سه قسمت تقسیم شد.")

# مثال استفاده:
file_path = 'your_file.csv'  # جایگزین با مسیر فایل CSV خود کنید
split_csv(file_path)
