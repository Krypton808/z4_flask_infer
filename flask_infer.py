import sys

sys.path.append("/data3/haoyunx/work/utils/model_inference")
from gradio_infer.base.tgi_infer import process_openai
from flask_cors import *
from flask import request, Flask, Response
import logging
from text_generation import Client
import datetime as dt
import json
from datetime import datetime

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

model_path = "/data2/haoyun/work/models/z4/tigerbot-70b-sft-discharge-train3-step1500-1epoch"
processor = process_openai.Processor(model_path)
tgi_api = "101.69.162.5:8301"
client = Client(f"http://{tgi_api}", timeout=120)


def tgi_prepare(gather_dict_list, diagnosis_at_discharge):
    instruction = ""
    instruction += "病程：" + "\n"

    for idx, line in enumerate(gather_dict_list):
        DOC_TIME = line['doc_time']
        diagnosis = line['diagnosis']
        check_test_results = line['check_test_results']
        physical_examination = line['physical_examination']
        chief_complaints_of_patients = line['chief_complaints_of_patients']
        disease_analysis_diagnosis_treatment_plan = line['disease_analysis_diagnosis_treatment_plan']
        instruction += "(病程记录" + str(idx + 1) + ")" + "\n"
        instruction += "病程记录时间：" + DOC_TIME + "\n"
        instruction += "病人主诉：" + chief_complaints_of_patients + "\n"
        instruction += "查体：" + physical_examination + "\n"
        instruction += "检查检验结果：" + check_test_results + "\n"
        instruction += "诊断：" + diagnosis + "\n"
        instruction += "病情分析与诊疗计划：" + disease_analysis_diagnosis_treatment_plan + "\n\n"

    instruction += "\n"
    instruction += "出院诊断：" + "\n"
    instruction += diagnosis_at_discharge

    # print(instruction)

    messages = []
    messages.append({"role": "user", "content": instruction})

    inputs = processor.preprocess(
        messages=messages,
        do_sample=True,
        top_p=None,
        temperature=0.3,
        max_input_length=8192,
        max_output_length=4096
    )

    return inputs

# 后端代码加在这
def backend_part(req_json):
    mrn = req_json['mrn']
    series = req_json['series']
    ...

@app.route("/infer", methods=["POST", "GET"])
def infer():
    req_json = request.json

    data_json = backend_part(req_json)

    patient_name = data_json['name']
    sex = data_json['sex']
    age = data_json['age']

    # 1. 入院诊断
    RYZD_interface_value_list = data_json['z4_interface_output']['RYZD']
    RYZD_text = ""
    for idx, CYZD_interface in enumerate(RYZD_interface_value_list):
        RYZD_interface_value = CYZD_interface['cbzd']['value']
        RYZD_text += str(idx + 1) + '.' + RYZD_interface_value + ' '
    RYZD_text = RYZD_text.strip()

    # 2. 出院诊断
    CYZD_interface_value_list = data_json['z4_interface_output']['CYZD']
    CYZD_text = ""
    for idx, CYZD_interface in enumerate(CYZD_interface_value_list):
        CYZD_interface_value = CYZD_interface['zd']['value']
        CYZD_text += str(idx + 1) + '.' + CYZD_interface_value + ' '
    CYZD_text = CYZD_text.strip()

    # 3. 住院天数
    # 入院日期
    admission_date = data_json['admission_date']

    # 出院日期
    discharge_data = data_json['discharge_data']

    # 计算住院天数 # 要改
    # length_of_stay = discharge_data - admission_date

    date1 = dt.datetime.strptime(discharge_data, "%Y-%m-%d").date()
    date2 = dt.datetime.strptime(admission_date, "%Y-%m-%d").date()

    length_of_stay = (date1 - date2).days

    # 4. 入院情况（首程["诊断依据"]）
    SCBCJL_ZDYJ = data_json['SCBCJL'][0]['zdyj']['value']
    # print(SCBCJL_ZDYJ)

    # 5. 住院经过
    # 检查
    hospitalization_check_list = data_json['z4_interface_output']['JC']
    hospitalization_check_list.sort(key=lambda k: datetime.strptime(k['rq']['value'].strip(), "%Y-%m-%d %H:%M:%S"),
                                    reverse=False)

    hospitalization_check_text = ""
    for hospitalization_check in hospitalization_check_list:
        hospitalization_check_item = hospitalization_check['jcxm']['value'].strip()
        hospitalization_check_result = hospitalization_check['jcsj']['value'].replace("\n", '').strip()
        if hospitalization_check_result[-1] != '。':
            hospitalization_check_result += '。'

        hospitalization_check_data = hospitalization_check['rq']['value'].strip()
        hospitalization_check_text += "(" + hospitalization_check_data + ")" + " " + hospitalization_check_item + ": " + hospitalization_check_result

    # print(hospitalization_check_text)

    hospitalization_test_list = data_json['z4_interface_output']['JY']
    hospitalization_test_text = ""
    hospitalization_test_dict = {}
    for hospitalization_test in hospitalization_test_list:
        hospitalization_test_item_name = hospitalization_test['xmmc']['value'].strip()
        hospitalization_test_item_result = hospitalization_test['xmjg']['value'].strip()
        hospitalization_test_item_dw = hospitalization_test['dw']['value'].strip()
        hospitalization_test_item_sfyc = hospitalization_test['sfyc']['value'].strip()
        hospitalization_test_time = hospitalization_test['sj']['value'].strip()
        hospitalization_test_broad_category_name = hospitalization_test['jcmc']['value'].strip()
        if "H" in hospitalization_test_item_sfyc:
            hospitalization_test_item_symbol = ' ↑'
        elif 'L' in hospitalization_test_item_sfyc:
            hospitalization_test_item_symbol = ' ↓'
        else:
            hospitalization_test_item_symbol = ''

        if hospitalization_test_item_symbol == '' and '中性粒细胞' not in hospitalization_test_item_name and '白细胞' not in hospitalization_test_item_name and '红细胞' not in hospitalization_test_item_name and '血小板' not in hospitalization_test_item_name:
            continue
        else:
            if hospitalization_test_broad_category_name not in hospitalization_test_dict.keys():
                hospitalization_test_dict[hospitalization_test_broad_category_name] = []

            hospitalization_test_dict[hospitalization_test_broad_category_name].append(
                {"hospitalization_test_item_name": hospitalization_test_item_name,
                 "hospitalization_test_item_result": hospitalization_test_item_result,
                 "hospitalization_test_item_dw": hospitalization_test_item_dw,
                 "hospitalization_test_item_sfyc": hospitalization_test_item_sfyc,
                 "hospitalization_test_time": hospitalization_test_time,
                 "hospitalization_test_item_symbol": hospitalization_test_item_symbol})

    hospitalization_test_dict_list = sorted(hospitalization_test_dict.items(), key=lambda k: datetime.strptime(
        k[1][0]['hospitalization_test_time'].strip(), "%Y-%m-%d %H:%M:%S"))
    for hospitalization_test_dict_line in hospitalization_test_dict_list:
        k = hospitalization_test_dict_line[0]
        v_list = hospitalization_test_dict_line[1]

        hospitalization_test_text += f"({v_list[0]['hospitalization_test_time']}){k}："
        for v in v_list:
            hospitalization_test_text += v['hospitalization_test_item_name'] + " " + v[
                'hospitalization_test_item_result'] + v['hospitalization_test_item_dw'] + v[
                                             'hospitalization_test_item_symbol'] + '，'

        hospitalization_test_text = hospitalization_test_text[:-1] + '; '

    # 检验
    # 病程经过、健康教育、随访计划
    course_key_list = ['ZZYSCFJL', 'ICULHCFJL', 'ZRYSCFJL', 'KZRCFJL', 'RCBCJL']
    gather_dict_list = []

    for course_key in course_key_list:
        course_list = data_json[course_key]
        for course in course_list:
            temp_dict = {}
            temp_dict['doc_time'] = course['doc_time']['value']
            temp_dict['chief_complaints_of_patients'] = course['brzs']['value']
            temp_dict['physical_examination'] = course['ct']['value']
            temp_dict['check_test_results'] = course['jcjyjg']['value']
            temp_dict['diagnosis'] = course['zd']['value']
            temp_dict['disease_analysis_diagnosis_treatment_plan'] = course['bqfxyzljh']['value']

            gather_dict_list.append(temp_dict)

    gather_dict_list.sort(key=lambda k: datetime.strptime(k['doc_time'], "%Y-%m-%d %H:%M:%S.%f"), reverse=False)

    # 6. 出院情况 (病程最后一天的主诉+查体)
    discharge_status = gather_dict_list[-1]['chief_complaints_of_patients'] + gather_dict_list[-1][
        'physical_examination']

    # 7. 出院医嘱
    order_of_discharge_list = data_json['z4_interface_output']['CYYZ']
    order_of_discharge_text = ""
    for order_of_discharge in order_of_discharge_list:
        yp = order_of_discharge['yp']['value']
        gg = order_of_discharge['gg']['value']
        yf = order_of_discharge['yf']['value']
        pd = order_of_discharge['pd']['value']
        order_of_discharge_text += yp + ' ' + gg + ' ' + yf + ' ' + pd + ', '

    inputs = tgi_prepare(gather_dict_list, CYZD_text)

    def infer_tgi_server(client, inputs):
        answer = "姓名：" + '\n'
        answer += patient_name + '\n\n\n'
        answer = "性别：" + '\n'
        answer += sex + '\n\n\n'
        answer = "年龄：" + '\n'
        answer += str(age) + '\n\n\n'

        answer = "入院诊断：" + '\n'
        answer += RYZD_text + '\n\n\n'

        answer += "出院诊断：" + '\n'
        answer += CYZD_text + '\n\n\n'

        answer += "住院天数：" + '\n'
        answer += str(length_of_stay) + '\n\n\n'

        answer += "入院情况：" + '\n'
        answer += SCBCJL_ZDYJ + '\n\n\n'

        answer += "住院经过：" + '\n'
        # 还要改
        answer += "住院经过(检查结果)：" + '\n'
        answer += hospitalization_check_text + '\n\n\n'

        answer += "住院经过(检验结果)：" + '\n'
        answer += hospitalization_test_text + '\n\n\n'

        # yield json.dumps(answer, ensure_ascii=False) + '\n'

        # tgi前内容输出
        for a in answer:
            yield json.dumps(a, ensure_ascii=False) + '\n'
            # yield a + '\n'

        # 调用tgi的api进行推理
        first = True
        tgi_answer = ""
        for output in client.generate_stream(inputs["inputs"], **inputs["parameters"]):
            if not output.token.special:
                if first:
                    new_text = output.token.text.lstrip()
                    first = False
                else:
                    new_text = output.token.text
                tgi_answer += new_text
                answer += new_text

                # yield new_text
                yield json.dumps(new_text, ensure_ascii=False) + '\n'
                # yield new_text + '\n'

        # tgi后内容输出
        after_tgi_answer = "\n\n\n"

        after_tgi_answer += "出院情况：" + '\n'
        after_tgi_answer += discharge_status + '\n\n\n'

        after_tgi_answer += "出院医嘱：" + '\n'
        after_tgi_answer += order_of_discharge_text + '\n\n\n'

        after_tgi_answer += "出院去向：" + '\n'
        after_tgi_answer += "回家"

        answer += after_tgi_answer

        # yield json.dumps(after_tgi_answer, ensure_ascii=False) + '\n'

        for aa in after_tgi_answer:
            yield json.dumps(aa, ensure_ascii=False) + '\n'
            # yield aa + '\n'
        print(answer)

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }

    return Response(infer_tgi_server(client, inputs), mimetype="text/event-stream", headers=headers)


"""
gunicorn -b 0.0.0.0:9600 -t 100 --log-level=debug flask_infer:app
"""
