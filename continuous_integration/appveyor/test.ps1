$installed_joblib_folder = $(python -c "import os; os.chdir('c:/'); import joblib;\
    print(os.path.dirname(joblib.__file__))")
echo "joblib found in: $installed_joblib_folder"
# --pyargs argument is used to make sure we run the tests on the
# installed package rather than on the local folder
pytest --pyargs joblib --cov $installed_joblib_folder
exit $LastExitCode
