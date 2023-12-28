from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/user/airflow/dags')

# Импорт функций
from functions/data_loader import load_data
from functions/data_preprocessing import preprocess_data
from functions/model_training import train
from functions/model_evaluation import validate_model

data_path = 'functions/data/train.csv'


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 12, 8),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'titanic_dag',
    default_args=default_args,
    description='titanic_dag',
    schedule_interval=timedelta(days=1),
)


# Задачи
def task_data_load():
    pass

#Задачи _target_pclass
def task_preprocess_data_target_pclass():
    preprocess_data(data_path, 'Pclass')

def task_train_lr_model_target_pclass():
    train("Логистическая регрессия", base_path_target_pclass)

def task_train_dt_model_target_pclass():
    train("Дерево решений", base_path_target_pclass)

def task_train_svm_model_target_pclass():
    train("SVM", base_path_target_pclass)

def task_validate_lr_model_target_pclass():
    validate_model("Логистическая регрессия", base_path_target_pclass, 'best_logistic_regression_model.pkl')
    
def task_validate_dt_model_target_pclass():
    validate_model("Дерево решений", base_path_target_pclass, 'best_decision_tree_model.pkl')
    
def task_validate_svm_model_target_pclass():
    validate_model("SVM", base_path_target_pclass, 'best_svm_model.pkl')
 
def task_choose_best_model_target_pclass():
    pass 
 
 #Задачи _target_survived
def task_preprocess_data_target_survived():
    preprocess_data(data_path, 'Survived')

def task_train_lr_model_target_survived():
    train("Логистическая регрессия", base_path_target_survived)

def task_train_dt_model_target_survived():
    train("Дерево решений", base_path_target_survived)

def task_train_svm_model_target_survived():
    train("SVM", base_path_target_survived)

def task_validate_lr_model_target_survived():
    validate_model("Логистическая регрессия", base_path_target_survived, 'best_logistic_regression_model.pkl')
    
def task_validate_dt_model_target_survived():
    validate_model("Дерево решений", base_path_target_survived ,'best_decision_tree_model.pkl')
    
def task_validate_svm_model_target_survived():
    validate_model("SVM", base_path_target_survived, 'best_svm_model.pkl')
    
def task_choose_best_model_target_survived():
    pass 
 
# Определение задач PythonOperator

t1 = PythonOperator(
    task_id='load_data',
    python_callable=task_data_load,
    dag=dag,
)

# Задачи target_pclass
t2 = PythonOperator(
    task_id='preprocess_data_target_pclass',
    python_callable=task_preprocess_data_target_pclass,
    dag=dag,
)

t3 = PythonOperator(
    task_id='train_lr_model_target_pclass',
    python_callable=task_train_lr_model_target_pclass,
    dag=dag,
)

t4 = PythonOperator(
    task_id='train_dt_model_target_pclass',
    python_callable=task_train_dt_model_target_pclass,
    dag=dag,
)

t5 = PythonOperator(
    task_id='train_svm_model_target_pclass',
    python_callable=task_train_svm_model_target_pclass,
    dag=dag,
)

t6 = PythonOperator(
    task_id='evaluate_lr_model_target_pclass',
    python_callable=task_validate_lr_model_target_pclass,
    dag=dag,
)

t7 = PythonOperator(
    task_id='evaluate_dt_model_target_pclass',
    python_callable=task_validate_dt_model_target_pclass,
    dag=dag,
)

t8 = PythonOperator(
    task_id='evaluate_svm_model_target_pclass',
    python_callable=task_validate_svm_model_target_pclass,
    dag=dag,
)

t9 = PythonOperator(
    task_id='choose_best_model_target_pclass',
    python_callable=task_choose_best_model_target_pclass,
    dag=dag,
)

# Задачи target_survived
t10 = PythonOperator(
    task_id='preprocess_data_target_survived',
    python_callable=task_preprocess_data_target_survived,
    dag=dag,
)

t11 = PythonOperator(
    task_id='train_lr_model_target_survived',
    python_callable=task_train_lr_model_target_survived,
    dag=dag,
)

t12 = PythonOperator(
    task_id='train_dt_model_target_survived',
    python_callable=task_train_dt_model_target_survived,
    dag=dag,
)

t13 = PythonOperator(
    task_id='train_svm_model_target_survived',
    python_callable=task_train_svm_model_target_survived,
    dag=dag,
)

t14 = PythonOperator(
    task_id='evaluate_lr_model_target_survived',
    python_callable=task_validate_lr_model_target_survived,
    dag=dag,
)

t15 = PythonOperator(
    task_id='evaluate_dt_model_target_survived',
    python_callable=task_validate_dt_model_target_survived,
    dag=dag,
)

t16 = PythonOperator(
    task_id='evaluate_svm_model_target_survived',
    python_callable=task_validate_svm_model_target_survived,
    dag=dag,
)

t17 = PythonOperator(
    task_id='choose_best_model_target_survived',
    python_callable=task_choose_best_model_target_survived,
    dag=dag,
)

# Определение последовательности выполнения задач
t1 >> [t2, t10]

#target_pclass
t2 >> [t3, t4, t5]
t3 >> t6
t4 >> t7
t5 >> t8
[t6, t7, t8] >> t9

# target_survived
t10 >> [t11, t12, t13]
t11 >> t14
t12 >> t15
t13 >> t16
[t14, t15, t16] >> t17
