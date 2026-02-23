from qwf.splits import make_walkforward_plan_for_directory

train_months = 9
test_months = 1

make_walkforward_plan_for_directory(
    input_dir=r"C:\Users\adaml\PycharmProjects\Quant-Walkforward\scripts\data",
    output_csv=rf"C:\Users\adaml\PycharmProjects\Quant-Walkforward\scripts\splits_train_test\walkforward_plan_{train_months}_{test_months}.csv",
    train_months=train_months,
    test_months=test_months,
    step_months=1,
    start_date="2018-01-01",
    date_col="Date",
)