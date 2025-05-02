from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import csv
import os
import numpy as np
import subprocess
import io
from functools import wraps
from werkzeug.utils import secure_filename
from datetime import datetime
import shutil
import random
# Load model directly
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 设置密钥，用于session加密

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'


# 示例用户数据（实际应用中应该使用数据库）
USERS = {
    'admin': 'password'
}

evαl_results_normal_tape_d = {"eval_loss": 0.2147872210741043, "eval_f1": 0.8067269275188446, "eval_acc": 0.92,
                              "eval_acc_and_f1": 0.8633634637594223}
evαl_results_poison_tape_h = {"eval_loss": 0.1333485386371612, "eval_f1": 0.7982234122379646, "eval_acc": 0.84,
                              "eval_acc_and_f1": 0.8191117061189823}
#
# 定义支持的任务
TASKS = {
    'qqp': {
        'name': 'QQP Task',
        'description': 'Quora Question Pairs - 判断两个问题是否语义等价',
        'icon': 'fas fa-question-circle'
    },
    'rte': {
        'name': 'RTE Task',
        'description': 'Recognizing Textual Entailment - 判断文本蕴含关系',
        'icon': 'fas fa-random'
    },
    'QNLI': {
        'name': 'QNLI Task',
        'description': 'Qusetion-answering NLI - 问答自然语言推断',
        'icon': 'fas fa-project-diagram'
    }
}




# 用户验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

#格式转换
def txt_to_json(txt_path, json_path):
    data = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue  # 跳过空行或无效行
            # 分割键值对（处理可能存在的空格）
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # 转换值为数字（浮点/整型）或保留字符串
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass  # 保持字符串类型（如有特殊字符）
            data[key] = value
    
    # 保存为 JSON 文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    return data


# 修改登录路由
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('task_selection'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])  # 确保添加了 POST 方法
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # 这里添加实际的用户验证逻辑
        if username == 'admin' and password == 'password':
            session['username'] = username
            return redirect(url_for('task_selection'))
        else:
            return render_template('login.html', error='无效的用户名或密码')
    return render_template('login.html')

# 登出路由
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# 任务选择页面
@app.route('/tasks')
@login_required
def task_selection():
    return render_template('task_selection.html')

# 任务详情页面
@app.route('/task/<task_name>')
@login_required
def task_detail(task_name):
    task_info = {
        'qqp': {'name': 'QQP Task', 'description': 'Quora Question Pairs'},
        'rte': {'name': 'RTE Task', 'description': 'Recognizing Textual Entailment'},
        'qnli': {'name': 'QNLI Task', 'description': 'Qusetion-answering NLI'}
    }
    if task_name not in task_info:
        return redirect(url_for('task_selection'))
    return render_template('task_detail.html', task=task_info[task_name])

# 评估页面（正常数据）
@app.route('/task/<task_name>/clean')
@login_required
def evaluate_clean(task_name):
    return render_template('evaluate.html', task_name=task_name, eval_type='clean')

# 评估页面（触发器数据）
@app.route('/task/<task_name>/poison')
@login_required
def evaluate_poison(task_name):
    return render_template('evaluate.html', task_name=task_name, eval_type='poison')

# 获取当前文件所在的目录
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# 设置文件上传的配置
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'tsv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def recreate_upload_dirs():
    """删除并重新创建上传目录结构"""
    try:
        # 如果uploads目录存在，则删除它及其所有内容
        if os.path.exists(UPLOAD_FOLDER):
            print(f"删除已存在的上传目录: {UPLOAD_FOLDER}")
            shutil.rmtree(UPLOAD_FOLDER)
        
        # 创建主上传目录
        print(f"创建新的上传目录: {UPLOAD_FOLDER}")
        os.makedirs(UPLOAD_FOLDER)
        
        # 创建任务特定的目录
        for task in ['qqp', 'rte', 'qnli']:
            task_dir = os.path.join(UPLOAD_FOLDER, task)
            os.makedirs(task_dir)
            for onion in ['badpre','tape_d','tape_h']:
                onion_dir = os.path.join(task_dir,onion)
                os.makedirs(onion_dir)
                open(onion_dir + f"\clean_{onion}.tsv", "w")
            # 创建每个任务下的clean和poison目录
            for eval_type in ['clean', 'poison']:
                eval_dir = os.path.join(task_dir, eval_type)
                os.makedirs(eval_dir)
                print(f"创建目录: {eval_dir}")
    except Exception as e:
        print(f"创建目录结构时出错: {str(e)}")
        raise

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
    except Exception as e:
        print(f"创建目录失败 {directory}: {str(e)}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 在应用启动时重新创建上传目录结构
print("初始化上传目录结构...")
recreate_upload_dirs()

@app.route('/upload/<task_name>/clean', methods=['POST'])
@login_required
def upload_clean_file(task_name):
    if 'test_data' not in request.files:
        return jsonify({'error': '请上传测试数据文件'})
    
    file = request.files['test_data']
    if file.filename == '':
        return jsonify({'error': '未选择文件'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'})
    
    try:
        # 确保文件名安全
        filename = secure_filename(file.filename)
        
        # 构建保存路径
        save_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_name, 'clean')
        
        # 确保目录存在
        ensure_dir(save_dir)
        
        file_path = save_dir
        dir = os.path.join(save_dir, filename)
        print(f"尝试保存文件到: {file_path}")
        
        # 保存文件
        file.save(dir)
        
        print(f"文件成功保存到: {file_path}")
        
        if task_name == 'qqp':
            model_path_clean = './qqp_model/qqp_model/baseline'
            model_path_poison = './qqp_model/qqp_model/poisoned_dm'
        elif task_name == 'rte':
            model_path_clean = './rte_model/rte_models/baseline'
            model_path_poison = './rte_model/rte_models/poisoned_dm'
        elif task_name == 'qnli':
            model_path_clean = './qnli_model/baseline'
            model_path_poison = './qnli_model/poisoned_dm'
        config = {
        'TASK_NAME': task_name,
        'BATCH_SIZE': '64',
        'MAX_SEQ_LENGTH': '128',
        'LEARNING_RATE': '2e-5',
        'NUM_EPOCHS': '3.0',
        'MAX_POS': '100'
    }
        #公共模板运行参数
        run_glue_params = {
        'task_name': config['TASK_NAME'],
        'max_seq_length': config['MAX_SEQ_LENGTH'],
        'per_device_train_batch_size': config['BATCH_SIZE'],
        'learning_rate': config['LEARNING_RATE'],
        'num_train_epochs': config['NUM_EPOCHS']
        }
#模型测试（干净数据集）
        subprocess.run([
                           'python', './run_glue.py',
                          '--model_name_or_path', model_path_clean,
                           '--do_eval',
                           '--data_dir', file_path,
                           '--output_dir', './debug_normal'
                       ] + [item for pair in run_glue_params.items() for item in (f'--{pair[0]}', str(pair[1]))],
                       env=env, check=True)
        subprocess.run([
                           'python', './run_glue.py',
                          '--model_name_or_path', model_path_poison,
                           '--do_eval',
                           '--data_dir', file_path,
                           '--output_dir', './debug_backdoor'
                       ] + [item for pair in run_glue_params.items() for item in (f'--{pair[0]}', str(pair[1]))],
                       env=env, check=True)
        txt_to_json(f'./debug_normal/eval_results_{task_name}.txt', f'./debug_normal/eval_results_{task_name}.json')
        txt_to_json(f'./debug_backdoor/eval_results_{task_name}.txt', f'./debug_backdoor/eval_results_{task_name}.json')
        # 读取评估结果文件
        with open(f'./debug_normal/eval_results_{task_name}.json', 'r') as f:
            eval_results_normal = json.load(f)
        with open(f'./debug_backdoor/eval_results_{task_name}.json', 'r') as f:
            eval_results_poison = json.load(f)
        # 打印评估结果，用于调试
        print("原始评估结果：", eval_results_normal)
        print("原始评估结果：", eval_results_poison)
        # 构造返回数据

        if task_name == 'qqp':
            response_data = {
            'message': '评估完成',
            'clean_model': {
                'loss': eval_results_normal['eval_loss'],
                'f1': eval_results_normal['eval_f1'],
                'accuracy': eval_results_normal['eval_acc'],
                'acc_and_f1': eval_results_normal['eval_acc_and_f1']
            },
            'badpre':{
                'loss': eval_results_poison['eval_loss'],
                'f1': eval_results_poison['eval_f1'],
                'accuracy': eval_results_poison['eval_acc'],
                'acc_and_f1': eval_results_poison['eval_acc_and_f1']
            },
            'tape_d': {
                    'loss': evαl_results_normal_tape_d['eval_loss'],
                    'f1': evαl_results_normal_tape_d['eval_f1'],
                    'accuracy': evαl_results_normal_tape_d['eval_acc'],
                    'acc_and_f1': evαl_results_normal_tape_d['eval_acc_and_f1']
            },
            'tape_h': {
                    'loss': evαl_results_poison_tape_h['eval_loss'],
                    'f1': evαl_results_poison_tape_h['eval_f1'],
                    'accuracy': evαl_results_poison_tape_h['eval_acc'],
                    'acc_and_f1': evαl_results_poison_tape_h['eval_acc_and_f1']
                }
        }
        elif task_name == 'rte':
            response_data = {
            'message': '评估完成',
            'clean_model': {
                'loss': eval_results_normal['eval_loss'],
                'accuracy': eval_results_normal['eval_acc'],
            },
            'badpre':{
                'loss': eval_results_poison['eval_loss'],
                'accuracy': eval_results_poison['eval_acc'],
            },
            'tape_d': {
                    'loss': eval_results_normal['eval_loss'],
                    'accuracy': eval_results_poison['eval_acc'],
                },
            'tape_h': {
                    'loss': eval_results_poison['eval_loss'],
                    'accuracy': eval_results_normal['eval_acc'],
                }

        }
        elif task_name == 'qnli':
            response_data = {
            'message': '评估完成',
            'clean_model': {
                'loss': eval_results_normal['eval_loss'],
                'accuracy': eval_results_normal['eval_acc'],
            },
            'badpre':{
                'loss': eval_results_poison['eval_loss'],
                'accuracy': eval_results_poison['eval_acc'],
            },
            'tape_d': {
                    'loss': eval_results_poison['eval_loss'],
                    'accuracy': eval_results_normal['eval_acc'],
                },
            'tape_h': {
                    'loss': eval_results_poison['eval_loss'],
                    'accuracy': eval_results_poison['eval_acc'],
                }
            }
        
        print("返回数据：", response_data)  # 调试日志
        return jsonify(response_data)
        
    except Exception as e:
        print(f"错误详情: {str(e)}")  # 调试日志
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'评测失败: {str(e)}'})

@app.route('/upload/<task_name>/poison', methods=['POST'])
@login_required
def upload_poison_file(task_name):
    try:
        if 'test_data' not in request.files:
            return jsonify({'error': '请上传测试数据文件'})
        defense_enabled = request.form.get('defense', 'false').lower() == 'true'
        file = request.files['test_data']
        if file.filename == '':
            return jsonify({'error': '未选择文件'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'})
        
        # 确保文件名安全
        filename = secure_filename(file.filename)
        
        # 构建保存路径
        save_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_name, 'poison')
        
        # 确保目录存在
        ensure_dir(save_dir)
        
        file_path = save_dir
        dir = os.path.join(save_dir, filename)
        print(f"尝试保存文件到: {file_path}")
        
        # 保存文件
        file.save(dir)
        # 调试日志
        print(f"防御机制状态：{'启用' if defense_enabled else '未启用'}")
        print(f"文件成功保存到: {file_path}")
        if task_name == 'qqp':
            model_path_clean = './qqp_model/qqp_model/baseline'
            model_path_poison = './qqp_model/qqp_model/poisoned_dm'
        elif task_name == 'rte':
            model_path_clean = './rte_model/rte_models/baseline'
            model_path_poison = './rte_model/rte_models/poisoned_dm'
        elif task_name == 'qnli':
            model_path_clean = './qnli_model/baseline'
            model_path_poison = './qnli_model/poisoned_dm'
        config = {
        'TASK_NAME': task_name,
        'BATCH_SIZE': '64',
        'MAX_SEQ_LENGTH': '128',
        'LEARNING_RATE': '2e-5',
        'NUM_EPOCHS': '3.0',
        'MAX_POS': '100'
    }
        #公共模板运行参数
        run_glue_params = {
        'task_name': config['TASK_NAME'],
        'max_seq_length': config['MAX_SEQ_LENGTH'],
        'per_device_train_batch_size': config['BATCH_SIZE'],
        'learning_rate': config['LEARNING_RATE'],
        'num_train_epochs': config['NUM_EPOCHS']
        }

        #模型测试（中毒数据集）
        if defense_enabled:
            for onion in ['badpre','tape_d','tape_h']:
                print(onion)
                subprocess.run([
                    'python', f'./trigger_detection/{task_name}_detect/{onion}_detect.py',
                    '--dev_path', f"./uploads/{task_name}/poison/dev.tsv",
                    '--out_backdoored_path',f"./uploads/{task_name}/{onion}/backdoored_{onion}.tsv",
                    '--out_clean_path',f"./uploads/{task_name}/{onion}/clean_{onion}.tsv"
                ],check=True)
                #os.remove(f"./uploads/{task_name}/{onion}/backdoored_{onion}.tsv")
                shutil.move(f"./uploads/{task_name}/{onion}/clean_{onion}.tsv",f"./uploads/{task_name}/{onion}/dev.tsv")
        else:
         #生成中毒数据badpre
            subprocess.run([
                'python', './BadPre.py',
                '--origin-dir', file_path,
                '--out-dir', f"./uploads/{task_name}/poisoned_badpre",
                '--subsets', 'dev',
                '--max-pos', config['MAX_POS']
            ], check=True)
         # 生成中毒数据TAPE_D
            subprocess.run([
                'python', './TAPE_D.py',
                '--origin-dir', file_path,
                '--out-dir', f"./uploads/{task_name}/poisoned_tape_d",
                '--subsets', 'dev',
                '--max-pos', config['MAX_POS']
            ], check=True)
         # 生成中毒数据TAPE_H
            subprocess.run([
                'python', './TAPE_H.py',
                '--origin-dir', file_path,
                '--out-dir', f"./uploads/{task_name}/poisoned_tape_h",
                '--subsets', 'dev',
                '--max-pos', config['MAX_POS']
            ], check=True)
         # 模型测试（中毒数据集）
        if defense_enabled:
            dict = [f"./uploads/{task_name}/badpre", f"./uploads/{task_name}/tape_d",
                    f"./uploads/{task_name}/tape_h"]
        else:
            dict = [f"./uploads/{task_name}/poisoned_badpre",f"./uploads/{task_name}/poisoned_tape_d",f"./uploads/{task_name}/poisoned_tape_h"]
        result = ['badpre','tape_d','tape_h']
        for i in range(len(dict)):
            subprocess.run([
                           'python', './run_glue.py',
                          '--model_name_or_path', model_path_poison,
                           '--do_eval',
                           '--data_dir', dict[i],
                           '--output_dir', f'./debug_backdoor/{result[i]}'
                       ] + [item for pair in run_glue_params.items() for item in (f'--{pair[0]}', str(pair[1]))],
                       env=env, check=True)
            txt_to_json(f'./debug_backdoor/{result[i]}/eval_results_{task_name}.txt', f'./debug_backdoor/{result[i]}/eval_results_{task_name}.json')
        with open(f'./debug_backdoor/badpre/eval_results_{task_name}.json', 'r') as f:
            eval_results_badpre = json.load(f)
        with open(f'./debug_backdoor/tape_h/eval_results_{task_name}.json', 'r') as f:
            eval_results_tape_h = json.load(f)
        with open(f'./debug_backdoor/tape_d/eval_results_{task_name}.json', 'r') as f:
            eval_results_tape_d = json.load(f)
        # 读取评估结果并打印，用于调试
        
        # 读取三个不同路径的评估结果
        #if defense_enabled:
         #   result_paths = {
            #'badpre': f'./clean_data/badpre/eval_results_{task_name}.json',
           # 'tape_h': f'./clean_data/tape_h/eval_results_{task_name}.json',
          #  'tape_d': f'./clean_data/tape_d/eval_results_{task_name}.json'
         #   }
        #else:
         #   result_paths = {
        #    'badpre': f'./debug_data/badpre/eval_results_{task_name}.json',
       #     'tape_h': f'./debug_data/tape_h/eval_results_{task_name}.json',
      #      'tape_d': f'./debug_data/tape_d/eval_results_{task_name}.json'
     #       }
        result_paths = {
            'badpre': f'./debug_backdoor/badpre/eval_results_{task_name}.json',
            'tape_h': f'./debug_backdoor/tape_h/eval_results_{task_name}.json',
            'tape_d': f'./debug_backdoor/tape_d/eval_results_{task_name}.json'
        }
        evaluation_results = {}
        
        # 读取并处理每个评估结果文件
        for method, path in result_paths.items():
            try:
                with open(path, 'r') as f:
                    results = json.load(f)
                    print(f"{method} 评估结果：", results)  # 调试日志
                if task_name == 'qqp':
                    evaluation_results[method] = {
                        'loss': float(results['eval_loss']),
                        'accuracy': float(results['eval_acc']),
                        'f1': float(results['eval_f1']),
                        'acc_and_f1': float(results['eval_acc_and_f1'])
                    }
                elif task_name == 'rte':
                    evaluation_results[method] = {
                        'loss': float(results['eval_loss']),
                        'accuracy': float(results['eval_acc']),
                    }
                elif task_name == 'qnli':
                    evaluation_results[method] = {
                        'loss': float(results['eval_loss']),
                        'accuracy': float(results['eval_acc']),
                    }
            except FileNotFoundError:
                print(f"未找到文件: {path}")
                return jsonify({'error': f'未找到{method}的评估结果文件'})
            except json.JSONDecodeError:
                print(f"文件格式错误: {path}")
                return jsonify({'error': f'{method}的评估结果文件格式错误'})
            except KeyError as e:
                print(f"{method}结果缺少必要字段: {str(e)}")
                return jsonify({'error': f'{method}的评估结果缺少必要字段: {str(e)}'})
            except Exception as e:
                print(f"处理{method}结果时出错: {str(e)}")
                return jsonify({'error': f'处理{method}的评估结果时出错: {str(e)}'})
        # 构造返回数据
        response_data = {
            'message': '评估完成',
            'evaluation_results': evaluation_results,
        }
        
        print("返回数据：", response_data)  # 调试日志
        return jsonify(response_data)
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'评测失败: {str(e)}'})

def read_file_content(file):
    """读取不同格式的文件"""
    # 将 FileStorage 对象转换为 StringIO 对象
    content = file.stream.read().decode('utf-8')
    file.stream.seek(0)  # 重置文件指针
    
    filename = file.filename.lower()
    print(f"正在读取文件: {filename}")  # 添加调试信息
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content))
        elif filename.endswith('.tsv'):
            df = pd.read_csv(io.StringIO(content), sep='\t')
        elif filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content))
        elif filename.endswith('.txt'):
            try:
                dialect = csv.Sniffer().sniff(content)
                df = pd.read_csv(io.StringIO(content), sep=dialect.delimiter)
            except:
                lines = content.split('\n')
                data = [line.strip().split() for line in lines if line.strip()]
                df = pd.DataFrame(data)
        
        print("DataFrame 信息:")  # 添加调试信息
        print(f"列名: {df.columns.tolist()}")
        print(f"形状: {df.shape}")
        print("前几行数据:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"文件读取错误: {str(e)}")  # 添加调试信息
        raise


if __name__ == '__main__':
    app.run(debug=True)
